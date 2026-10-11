"""Do the 101 Formulaic Alphas survive? Daily long-short portfolios on S&P 500 stocks, with multiple-testing
corrections and trading costs, before and after the paper appeared (December 2015).

Each alpha's signal at the close of day t is demeaned across stocks and scaled to a gross exposure of 1.
It is traded at the open of day t+1 and earns the open-to-open return to day t+2, so nothing uses data
that was not available. Cost = COST_BPS per unit of turnover.
"""
import glob, json, os, sys, time, warnings
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Data, evaluate
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
HERE = os.path.dirname(os.path.abspath(__file__))
COST_BPS = 5.0
SPLIT = "2016-01-01"
START = "2006-01-01"

members = pd.read_csv(f"{HERE}/data/members.csv").set_index("yahoo")
frames = {}
for path in sorted(glob.glob(f"{HERE}/data/prices/*.csv")):
    frames[os.path.basename(path)[:-4]] = pd.read_csv(path, parse_dates=["date"], index_col="date")
tickers = [t for t in members.index if t in frames]
dates = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
panels = {c: np.column_stack([frames[t][c].reindex(dates).values for t in tickers]).astype(float) for c in ("open", "high", "low", "close", "volume")}
panels["volume"][panels["volume"] <= 0] = np.nan
groups = {"sector": members.loc[tickers, "sector"].values, "sub_industry": members.loc[tickers, "sub_industry"].values}
data = Data(panels, groups)
print(f"{len(tickers)} stocks, {len(dates)} days, {dates[0].date()} to {dates[-1].date()}; stocks with data on the first day: {np.isfinite(panels['close'][0]).sum()}", flush=True)

op = panels["open"]
fwd = np.full_like(op, np.nan); fwd[:-2] = op[2:] / op[1:-1] - 1          # open(t+1) -> open(t+2)
formulas = json.load(open(f"{HERE}/data/formulas_raw.json"))
pnl, turnover, skipped = {}, {}, {}
t0 = time.time()
for n in range(1, 102):
    try:
        sig = evaluate(formulas[str(n)], data)
    except KeyError as e:
        skipped[n] = f"needs {e}"; continue
    sig = np.where(np.isfinite(fwd), sig, np.nan)
    w = sig - np.nanmean(sig, axis=1, keepdims=True)
    gross = np.nansum(np.abs(w), axis=1, keepdims=True)
    w = np.nan_to_num(np.where(gross > 0, w / gross, 0.0))
    if (np.abs(w).sum(1) > 0).mean() < 0.5:
        skipped[n] = "signal is empty on most days"; continue
    pnl[n] = np.nansum(w * np.nan_to_num(fwd), axis=1)
    turnover[n] = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, w.shape[1])))).sum(1)
print(f"evaluated {len(pnl)} alphas in {time.time() - t0:.0f}s; skipped: {skipped}", flush=True)

P = pd.DataFrame(pnl, index=dates)[START:].iloc[:-2]
TO = pd.DataFrame(turnover, index=dates)[START:].iloc[:-2]
NET = P - TO * COST_BPS / 1e4


def summarise(R):
    n = len(R); m, s = R.mean(), R.std(ddof=1)
    out = pd.DataFrame({"sharpe": m / s * np.sqrt(252), "t": m / (s / np.sqrt(n)), "sr_daily": m / s, "skew": R.skew(), "kurt": R.kurt() + 3, "n": n})
    out["p"] = stats.norm.sf(out.t)                                     # one-sided: the paper states each alpha's direction
    return out


def deflated(S):
    """Probability that each Sharpe ratio beats the best expected from this many trials of no skill (Bailey and Lopez de Prado)."""
    N, g = len(S), 0.5772156649
    sr0 = S.sr_daily.std(ddof=1) * ((1 - g) * stats.norm.ppf(1 - 1 / N) + g * stats.norm.ppf(1 - 1 / (N * np.e)))
    z = (S["sr_daily"] - sr0) * np.sqrt(S["n"] - 1) / np.sqrt(1 - S["skew"] * S["sr_daily"] + (S["kurt"] - 1) / 4 * S["sr_daily"] ** 2)
    return stats.norm.cdf(z), sr0 * np.sqrt(252)


rows, detail = [], {}
for period, sl in (("before the paper (2006-2015)", slice(START, "2015-12-31")), ("after the paper (2016-2026)", slice(SPLIT, None))):
    for label, R in (("before costs", P[sl]), (f"after {COST_BPS:g} bps costs", NET[sl])):
        S = summarise(R); N = len(S)
        dsr, bar = deflated(S); S["dsr"] = dsr
        ps = np.sort(S.p.values); bh = int(np.max(np.where(ps <= 0.10 * np.arange(1, N + 1) / N)[0], initial=-1) + 1)
        rows.append({"period": period, "returns": label, "alphas": N, "median Sharpe": round(S.sharpe.median(), 2), "Sharpe > 0": int((S.sharpe > 0).sum()),
                     "t > 2": int((S.t > 2).sum()), "t > 3": int((S.t > 3).sum()), "Bonferroni 5%": int((S.p < 0.05 / N).sum()),
                     "BH 10%": bh, "deflated Sharpe > 95%": int((S.dsr > 0.95).sum()), "luck's best Sharpe": round(bar, 2)})
        detail[period, label] = S
table = pd.DataFrame(rows)
print("\n" + table.to_string(index=False))
table.to_csv(f"{HERE}/survival_table.csv", index=False)

keys = list(detail)
b_g, b_n, a_g, a_n = (detail[k] for k in keys)
per = pd.DataFrame({"sharpe_before_gross": b_g.sharpe, "t_before_gross": b_g.t, "sharpe_before_net": b_n.sharpe, "t_before_net": b_n.t,
                    "sharpe_after_gross": a_g.sharpe, "t_after_gross": a_g.t, "sharpe_after_net": a_n.sharpe, "t_after_net": a_n.t,
                    "daily_turnover": TO.mean()}).round(3)
per.index.name = "alpha"; per.to_csv(f"{HERE}/per_alpha.csv")
print(f"\nmedian daily turnover {TO.mean().median():.2f} of gross exposure; expected false positives at t > 2 among {len(per)} with no edge: {len(per) * stats.norm.sf(2):.1f}")
print(f"Sharpe before costs: mean {b_g.sharpe.mean():.2f} before the paper, {a_g.sharpe.mean():.2f} after; correlation across alphas between the two periods {b_g.sharpe.corr(a_g.sharpe):.2f}")
both = per[(per.t_before_net > 2) & (per.t_after_net > 2)]
print(f"t > 2 after costs in BOTH periods: {len(both)} alphas {list(both.index)}")
strict = per[(per.t_before_net > 3) & (per.t_after_net > 3)]
print(f"t > 3 after costs in BOTH periods: {len(strict)} alphas {list(strict.index)}")
corr = P.corr().values; iu = np.triu_indices_from(corr, 1)
print(f"average absolute correlation between alphas' daily returns: {np.abs(corr[iu]).mean():.2f}")
print("\nbest ten by t after costs, after the paper:"); print(per.sort_values("t_after_net", ascending=False).head(10).to_string())
print("\nfive worst after costs, after the paper:"); print(per.sort_values("t_after_net").head(5)[["sharpe_after_gross", "sharpe_after_net", "daily_turnover"]].to_string())

# ---- how the count depends on the cost assumed, and on waiting one more day to trade
print("\nsurvivors by cost (t > 3 / Bonferroni 5%), one-way cost per unit of turnover:")
sens = []
for c in (0, 1, 2, 5, 10):
    row = {"cost_bps": c}
    for name, sl in (("before", slice(START, "2015-12-31")), ("after", slice(SPLIT, None))):
        S = summarise((P - TO * c / 1e4)[sl])
        row[f"{name}: t>3"] = int((S.t > 3).sum()); row[f"{name}: Bonferroni"] = int((S.p < 0.05 / len(S)).sum()); row[f"{name}: median Sharpe"] = round(S.sharpe.median(), 2)
    sens.append(row)
sens = pd.DataFrame(sens); print(sens.to_string(index=False)); sens.to_csv(f"{HERE}/cost_sensitivity.csv", index=False)
mean_gross, mean_to = P.mean(), TO.mean()
breakeven = (mean_gross / mean_to * 1e4)
print(f"break-even cost: median {breakeven.median():.1f} bps, 90th percentile {breakeven.quantile(0.9):.1f} bps, alpha 42: {breakeven[42]:.1f} bps")
fwd2 = np.full_like(op, np.nan); fwd2[:-3] = op[3:] / op[2:-1] - 1                          # trade one day later: open(t+2) -> open(t+3)
shifted = {}
for n in pnl:
    sig = evaluate(formulas[str(n)], data); sig = np.where(np.isfinite(fwd2), sig, np.nan)
    w = sig - np.nanmean(sig, axis=1, keepdims=True); g = np.nansum(np.abs(w), axis=1, keepdims=True)
    w = np.nan_to_num(np.where(g > 0, w / g, 0.0)); shifted[n] = np.nansum(w * np.nan_to_num(fwd2), axis=1)
L = pd.DataFrame(shifted, index=dates)[START:].iloc[:-3]
print(f"one extra day of delay, before costs: median Sharpe {summarise(L[:'2015-12-31']).sharpe.median():.2f} before the paper, {summarise(L[SPLIT:]).sharpe.median():.2f} after")
