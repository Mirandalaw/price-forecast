"""뉴스 사건 피처 (news_collector/scripts/run_agri_signals.py 출력 CSV → 모델 피처).

리키지 원칙: 예측 기준일 t 의 피처는 t 당일(발행시각 ≤ 마감)까지의 기사만. 여기서는 달력일 기준으로
t 이하를 쓰고, 발행시각 컷은 수집기(스냅샷 생성 시점)에서 보장한다.

생성 피처 (품목별):
  nw_cnt, nw_dir, nw_wdir           당일
  nw_dir_3d, nw_wdir_7d             3·7일 롤링 합
  nw_shock                          |nw_wdir| 이 과거 60일 분포의 90분위 이상이면 1
  nw_release, nw_buy, nw_import,
  nw_weather, nw_area, nw_volume    사건군별 7일 롤링 카운트
  mc_fx_7d, mc_oil_7d               거시 7일 롤링 (수입품에만 의미)
"""
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = {
    "nw_release": ["evt_agri_reserve_release", "evt_agri_discount_support", "evt_agri_volume_up"],
    "nw_buy": ["evt_agri_reserve_buy", "evt_agri_shipment_delay"],
    "nw_import": ["evt_agri_import_up"],
    "nw_weather": ["evt_agri_typhoon", "evt_agri_heavy_rain", "evt_agri_heat", "evt_agri_cold", "evt_agri_disease"],
    "nw_area": ["evt_agri_area_down", "evt_agri_good_crop"],
    "nw_demand": ["evt_agri_kimjang_demand"],
}
NEWS_FEATS = ["nw_cnt", "nw_dir", "nw_wdir", "nw_dir_3d", "nw_wdir_7d", "nw_shock",
              *GROUPS.keys(), "mc_fx_7d", "mc_oil_7d"]


def load_signals(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def daily_news_features(signals: pd.DataFrame, items: list[str], start, end) -> pd.DataFrame:
    """달력일 전체 격자(품목×날짜)에 신호를 놓고 롤링 피처를 만든다. 기사 없는 날 = 0."""
    idx = pd.date_range(start, end, freq="D")
    out = []
    for item in items:
        s = signals[signals["item"] == item].set_index("date").reindex(idx).fillna(0.0)
        f = pd.DataFrame(index=idx)
        f["item"] = item
        f["nw_cnt"] = s.get("news_cnt", 0.0)
        f["nw_dir"] = s.get("dir_sum", 0.0)
        f["nw_wdir"] = s.get("dir_wsum", 0.0)
        f["nw_dir_3d"] = f["nw_dir"].rolling(3, min_periods=1).sum()
        f["nw_wdir_7d"] = f["nw_wdir"].rolling(7, min_periods=1).sum()
        thr = f["nw_wdir"].abs().rolling(60, min_periods=10).quantile(0.9).shift(1)
        f["nw_shock"] = ((f["nw_wdir"].abs() >= thr) & (f["nw_wdir"] != 0)).astype(int)
        for g, cols in GROUPS.items():
            present = [c for c in cols if c in s.columns]
            f[g] = s[present].sum(axis=1).rolling(7, min_periods=1).sum() if present else 0.0
        f["mc_fx_7d"] = s.get("macro_fx_up", 0.0).rolling(7, min_periods=1).sum()
        f["mc_oil_7d"] = s.get("macro_oil_up", 0.0).rolling(7, min_periods=1).sum()
        out.append(f.rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True)


def attach(feat: pd.DataFrame, signals: pd.DataFrame | None) -> pd.DataFrame:
    """가격 피처 테이블(거래일)에 뉴스 피처를 붙인다. signals=None 이면 전부 0 (뉴스 off)."""
    feat = feat.copy()
    if signals is None or signals.empty:
        for c in NEWS_FEATS:
            feat[c] = 0.0
        return feat
    items = sorted(feat["item"].unique())
    nf = daily_news_features(signals, items, feat["date"].min() - pd.Timedelta(days=60), feat["date"].max())
    return feat.merge(nf, on=["item", "date"], how="left").fillna({c: 0.0 for c in NEWS_FEATS})
