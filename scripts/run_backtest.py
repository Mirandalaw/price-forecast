import sys
sys.path.insert(0, ".")
import argparse
from pathlib import Path
from price_forecast import backtest, loader, news

ap = argparse.ArgumentParser()
ap.add_argument("--signals", default="data/agri_signals.csv", help="news_collector run_agri_signals.py 출력. 없으면 뉴스 off")
a = ap.parse_args()
prices = loader.load_prices("data")
print(prices.groupby("item")["date"].agg(["min", "max", "count"]))
sig = news.load_signals(a.signals) if Path(a.signals).exists() else None
print("뉴스 피처:", "on" if sig is not None else "off (data/agri_signals.csv 없음)")
res = backtest.run(prices, signals=sig)
res.to_csv("data/backtest_results.csv", index=False)
print(backtest.report(res))
