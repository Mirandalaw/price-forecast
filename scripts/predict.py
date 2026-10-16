"""기준일 하나의 제출용 예측 파일 생성.

  python scripts/predict.py --asof 2024-10-26                      # output/forecast_2024-10-26.csv
  python scripts/predict.py --asof 2024-10-26 --model lgbm --signals data/agri_signals.csv
"""
import sys
sys.path.insert(0, ".")
import argparse
import logging
from pathlib import Path

from price_forecast import loader, news, predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
ap = argparse.ArgumentParser()
ap.add_argument("--asof", required=True, help="예측 기준일 YYYY-MM-DD")
ap.add_argument("--model", default=predict.DEFAULT_MODEL, choices=["persist", "drift_m", "lgbm", "blend", "lgbm_news", "blend_news"])
ap.add_argument("--signals", default="data/agri_signals.csv")
ap.add_argument("--out", default=None)
a = ap.parse_args()

prices = loader.load_prices("data")
sig = news.load_signals(a.signals) if Path(a.signals).exists() else None
fc = predict.forecast(prices, a.asof, signals=sig, model=a.model)
out = Path(a.out or f"output/forecast_{a.asof}.csv")
out.parent.mkdir(parents=True, exist_ok=True)
fc.to_csv(out, index=False)
print(fc.to_string(index=False))
print(f"\n→ {out}")
