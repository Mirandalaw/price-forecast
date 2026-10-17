"""시장 구조 피처 (시장별 KAMIS 자료에서만 나온다 — scripts/build_kamis_market.py 의 mk_* 열).

mk_qspread  상품/중품 로그 스프레드      mk_garak  가락/지방 로그 격차      mk_disp  시장 간 분산
mk_nkinds   조사 품종 수                mk_kindage 품종 구성이 바뀐 뒤 경과 조사일(작형 전환)
파생: *_d7 = 당일값 − 직전 7거래일 평균 (수준이 아니라 변화를 본다).
자료에 mk_* 열이 없으면(at5, 본선 경락가) 전부 NaN → LightGBM 이 결측 분기로 처리, 사실상 off.
"""
import numpy as np
import pandas as pd

MKT_BASE = ["mk_qspread", "mk_garak", "mk_disp", "mk_nkinds", "mk_kindage"]
MKT_FEATS = MKT_BASE + ["mk_qspread_d7", "mk_garak_d7", "mk_disp_d7"]


def add_features(feat: pd.DataFrame) -> pd.DataFrame:
    feat = feat.copy()
    if not all(c in feat.columns for c in MKT_BASE):
        for c in MKT_FEATS:
            feat[c] = np.nan
        return feat
    feat = feat.sort_values(["item", "date"])
    for c in ["mk_qspread", "mk_garak", "mk_disp"]:
        prev7 = feat.groupby("item")[c].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
        feat[f"{c}_d7"] = feat[c] - prev7
    return feat.sort_index()


def available(feat: pd.DataFrame) -> bool:
    return "mk_qspread" in feat.columns and feat["mk_qspread"].notna().any()
