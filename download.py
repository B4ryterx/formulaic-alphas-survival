"""Download daily prices for current S&P 500 members from Yahoo Finance's public chart service,
and the sector list from Wikipedia. Prices are adjusted for splits and dividends.

Known bias: only companies in the index today are included, so firms that failed or were dropped are missing.
Usage: download.py [start_year]
"""
import io, json, os, sys, time, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
START = int(sys.argv[1]) if len(sys.argv) > 1 else 2005
UA = {"User-Agent": "Mozilla/5.0"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()


html = get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").decode()
members = pd.read_html(io.StringIO(html))[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "Date added"]]
members.columns = ["symbol", "name", "sector", "sub_industry", "date_added"]
members["yahoo"] = members.symbol.str.replace(".", "-", regex=False)
members.to_csv(f"{HERE}/data/members.csv", index=False)
print(len(members), "members,", members.sector.nunique(), "sectors", flush=True)

p1, p2 = int(pd.Timestamp(f"{START}-01-01").timestamp()), int(time.time())
ok = failed = 0
for sym in members.yahoo:
    out = f"{HERE}/data/prices/{sym}.csv"
    if os.path.exists(out): ok += 1; continue
    for attempt in range(3):
        try:
            r = json.loads(get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d&events=div%2Csplits"))["chart"]["result"][0]
            q = r["indicators"]["quote"][0]
            d = pd.DataFrame({"date": pd.to_datetime(r["timestamp"], unit="s").normalize(), "open": q["open"], "high": q["high"], "low": q["low"],
                              "close": q["close"], "volume": q["volume"], "adjclose": r["indicators"]["adjclose"][0]["adjclose"]}).dropna()
            f = d.adjclose / d.close                       # one factor for all four prices keeps each day's shape
            for c in ("open", "high", "low", "close"): d[c] = (d[c] * f).round(4)
            d.drop(columns="adjclose").to_csv(out, index=False); ok += 1
            break
        except Exception as e:
            if attempt == 2: failed += 1; print("failed", sym, type(e).__name__, flush=True)
            time.sleep(2 + 3 * attempt)
    time.sleep(0.25)
    if (ok + failed) % 100 == 0: print(ok, "done", failed, "failed", flush=True)
print("finished:", ok, "saved,", failed, "failed")
