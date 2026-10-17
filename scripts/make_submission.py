"""회차 제출 파일 한 번에: 가격 로드 → (뉴스 신호) → 예측 → 검증·클리핑 → CSV + manifest.

  python scripts/make_submission.py --asof 2024-10-26
  python scripts/make_submission.py --asof 2024-10-26 --model lgbm_news --signals data/agri_signals.csv
종료 코드 0 = 제출 가능. 검증 실패(행 누락·NaN)는 예외로 종료 코드 1.
"""
import sys
sys.path.insert(0, ".")
import argparse
import logging
from pathlib import Path

from price_forecast import loader, news, predict, submission, weather

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
ap = argparse.ArgumentParser()
ap.add_argument("--asof", required=True)
ap.add_argument("--model", default="auto", help="auto = 품목별(배추·무 날씨 LightGBM, 저장품 기준가 유지)")
ap.add_argument("--signals", default="data/agri_signals.csv")
ap.add_argument("--data", default="data")
ap.add_argument("--out-dir", default="output")
a = ap.parse_args()

prices = loader.load_prices(a.data)
sig_path = Path(a.signals)
sig = news.load_signals(sig_path) if sig_path.exists() and a.model.endswith("_news") else None
wx = weather.load_daily(a.data)
fc = predict.forecast(prices, a.asof, signals=sig, model=a.model, weather=wx)
inputs = sorted(Path(a.data).glob("*.csv")) + ([sig_path] if sig is not None else [])
out, man = submission.write(fc, a.asof, a.model, inputs, Path(a.out_dir))
full = submission.validate(fc)
flagged = full[full["flag"] != ""]
print(full[["item", "horizon", "target", "base_price", "pred_price", "model", "flag"]].to_string(index=False))
print(f"\n제출 파일: {out}\n매니페스트: {man}\n클리핑 {len(flagged)}행 · 폴백 {int(full['model'].str.contains('fallback').sum())}행")
