import sys
sys.path.insert(0, ".")
import argparse
from pathlib import Path
from price_forecast import backtest, loader, news, weather

ap = argparse.ArgumentParser()
ap.add_argument("--signals", default="data/agri_signals.csv", help="news_collector run_agri_signals.py 출력. 없으면 뉴스 off")
ap.add_argument("--years", default="", help="평가 연도 쉼표 구분 (기본 2021~2024)")
a = ap.parse_args()
prices = loader.load_prices("data")
print(prices.groupby("item")["date"].agg(["min", "max", "count"]))
sig = news.load_signals(a.signals) if Path(a.signals).exists() else None
print("뉴스 피처:", "on" if sig is not None else "off (data/agri_signals.csv 없음)")
wx = weather.load_daily("data")
print("날씨 피처:", f"on ({wx['date'].min().date()}~{wx['date'].max().date()}, 지점 {sorted(wx['stn'].unique())})" if wx is not None else "off")
years = tuple(int(y) for y in a.years.split(",")) if a.years else (2021, 2022, 2023, 2024)
res = backtest.run(prices, years=years, signals=sig, weather=wx)
res.to_csv("data/backtest_results.csv", index=False)
print(backtest.report(res))
