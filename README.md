# Do the 101 Formulaic Alphas survive?

A test of whether the trading signals published in Kakushadze's *101 Formulaic Alphas* (2015) hold up
once you correct for how many were tried, subtract trading costs, and look at the years after publication.

## Result

100 of the 101 signals, traded as daily long-short portfolios on 503 S&P 500 stocks, 2006 to October 2026.

| Period | Returns | Median Sharpe | Sharpe > 0 | t > 2 | t > 3 | Bonferroni 5% | Benjamini-Hochberg 10% | Deflated Sharpe > 95% |
|---|---|---|---|---|---|---|---|---|
| Before the paper (2006-2015) | Before costs | 0.91 | 93 | 67 | 44 | 39 | 85 | 1 |
| Before the paper (2006-2015) | After 5 bps | -1.82 | 6 | 2 | 1 | 1 | 1 | 0 |
| After the paper (2016-2026) | Before costs | 0.49 | 89 | 38 | 14 | 10 | 50 | 0 |
| After the paper (2016-2026) | After 5 bps | -1.73 | 4 | 1 | 1 | 1 | 1 | 0 |

With no edge anywhere, about 2 of 100 would show t > 2.

- **Before costs the signals are real, not luck.** 39 pass Bonferroni before the paper, far more than chance allows.
- **They weakened after publication.** The mean Sharpe ratio fell from 0.87 to 0.48, and the number passing t > 3 fell from 44 to 14.
  Signals that were strong before stayed relatively strong (correlation of Sharpe ratios across the two periods: 0.55).
- **Costs remove almost all of them.** The median signal turns over 70% of its portfolio every day and breaks even at a
  cost of 1.2 bps per unit traded. At 5 bps, one signal survives in both periods.

| One-way cost (bps) | t > 3 before the paper | t > 3 after the paper |
|---|---|---|
| 0 | 44 | 14 |
| 1 | 14 | 2 |
| 2 | 4 | 1 |
| 5 | 1 | 1 |
| 10 | 0 | 0 |

- **The survivor is Alpha 42**, a one-day reversal signal with low turnover (34% a day): Sharpe 1.14 before and 1.09 after
  the paper, net of 5 bps, with t above 3.5 in both. Its break-even cost is about 19 bps. It depends on the price proxy noted below.
- **The deflated Sharpe ratio is the strictest test.** It asks whether a Sharpe ratio beats the best expected from 100 unskilled
  trials; one signal clears 95% before costs in the first period and none do after costs.

## Method

- **Formulas.** `extract_formulas.py` reads the 101 expressions from your own copy of the paper (arXiv:1601.00991).
  `engine.py` parses each one and evaluates it on arrays of days by stocks, using the operators defined in the paper's appendix.
- **Portfolio.** The signal at the close of day t is demeaned across stocks and scaled to gross exposure 1. It trades at the
  open of day t+1 and earns the open-to-open return to day t+2, so no signal uses information from after it was formed.
- **Costs.** A fixed cost per unit of turnover, 5 bps in the headline table.
- **Tests.** One-sided, because the paper states each signal's direction: t above 2 and above 3, Bonferroni at 5%,
  Benjamini-Hochberg at 10%, and the deflated Sharpe ratio of Bailey and López de Prado.

## Limits

- **Survivorship.** The universe is today's S&P 500 members, so companies that failed or were dropped are missing. Reversal
  signals buy recent losers, and here every loser is known to survive, which flatters results, especially in 2008-2009.
- **No intraday data.** VWAP is approximated by (high + low + close) / 3. Alpha 42, the survivor, is built on VWAP.
- **Industry levels.** Sector and sub-industry come from today's GICS labels; "industry" uses sub-industry.
- **Alpha 56 is skipped** because it needs market capitalisation.
- **Costs are a flat rate**, with no market impact, borrowing cost or short-sale constraint.
- **Trading one day later** lowers the median Sharpe ratio before costs from 0.91 to 0.67 (before) and 0.49 to 0.36 (after),
  so much of the effect is very short-lived.

## Run

```
python3 download.py 2005                        # prices from Yahoo Finance, sectors from Wikipedia
python3 extract_formulas.py 1601.00991.pdf      # your copy of the paper
python3 run.py                                  # tables above, per_alpha.csv, cost_sensitivity.csv
```

Needs Python 3 with numpy, pandas, scipy, bottleneck and pypdf. Price data and the formula text are not in the repository.
