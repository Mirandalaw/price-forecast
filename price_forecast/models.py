"""예측 모델 3종. 모두 '로그변화율 ŷ_h'를 내고, 가격은 P(t)·exp(ŷ_h).
h 는 1..10 (거래일 앞) 또는 "s2"/"s3" (2순·3순 후 순 평균) — 열 이름 규칙 y{h}/p{h} 가 같아서 코드가 공유된다.

persistence : ŷ = 0
drift       : ŷ = seasonal_drift (과거 연도 같은 시기 평균 변화율)
lgbm        : 예측일별 LightGBM 10개, L1 목적함수 + 1/실제가격 가중(MAPE 정합), 피처 = 래그·MA·평년비·drift
blend       : lgbm·drift 가중 평균 후 h가 멀수록 drift 쪽으로 수축, 품목별 과거 1%·99% 분위로 클리핑
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from .news import NEWS_FEATS
from .weather import WX_FEATS
from .market import MKT_FEATS

PRICE_FEATS = ["lag_ret1", "lag_ret2", "lag_ret3", "lag_ret5", "lag_ret7", "lag_ret14",
               "ma7_gap", "ma28_gap", "vol14", "dow", "doy", "sun_gap", "normal_gap", "ly_gap", "drift"]
FEATS = PRICE_FEATS  # 기본(뉴스 off). use_news=True 면 PRICE_FEATS + NEWS_FEATS
PARAMS = dict(objective="l1", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)


def feats(use_news: bool, use_wx: bool = False, use_mkt: bool = False) -> list[str]:
    return PRICE_FEATS + (NEWS_FEATS if use_news else []) + (WX_FEATS if use_wx else []) + (MKT_FEATS if use_mkt else [])


def _x(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """가격·뉴스 피처의 결측은 0, 날씨 결측은 NaN 그대로 (LightGBM 이 결측 분기를 학습)."""
    x = df[cols].copy()
    fill = [c for c in cols if c not in WX_FEATS and c not in MKT_FEATS]
    x[fill] = x[fill].fillna(0)
    return x


def shrink(h) -> float:
    """h가 멀수록 lgbm 비중을 drift 쪽으로 수축. 일별 h=1→1.0, h=10→0.64 / 순별 2순·3순 후→0.5."""
    return 0.5 if isinstance(h, str) else 1 - 0.04 * (h - 1)


def fit_lgbm(train: pd.DataFrame, h, use_news: bool = False, use_wx: bool = False, use_mkt: bool = False) -> lgb.Booster:
    F_ = feats(use_news, use_wx, use_mkt)
    t = train.dropna(subset=PRICE_FEATS + [f"y{h}", f"p{h}"])
    w = 1.0 / t[f"p{h}"].clip(lower=1)          # MAPE 가중: 실제가격이 낮을수록 오차가 비싸다
    w = w / w.mean()
    ds = lgb.Dataset(_x(t, F_), t[f"y{h}"], weight=w, categorical_feature=["dow"])
    return lgb.train(PARAMS, ds, num_boost_round=400)


def predict_all(train: pd.DataFrame, test: pd.DataFrame, h, w_lgbm: float = 0.6,
                use_news: bool = False, use_wx: bool = False, use_mkt: bool = False) -> pd.DataFrame:
    """test 행마다 persistence / drift / lgbm / blend 의 ŷ_h 를 돌려준다."""
    out = test[["item", "date", "price", f"p{h}", f"y{h}", "drift"]].copy()
    out["persist"] = 0.0
    out["drift_m"] = out["drift"].fillna(0.0)
    m = fit_lgbm(train, h, use_news, use_wx, use_mkt)
    cols = feats(use_news, use_wx, use_mkt)
    out["lgbm"] = np.where(test[PRICE_FEATS].isna().any(axis=1), out["drift_m"], m.predict(_x(test, cols)))
    sh = shrink(h)
    out["blend"] = (w_lgbm * sh) * out["lgbm"] + (1 - w_lgbm * sh) * out["drift_m"]
    lo = train.groupby("item")[f"y{h}"].quantile(0.01); hi = train.groupby("item")[f"y{h}"].quantile(0.99)
    out["blend"] = out.apply(lambda r: float(np.clip(r["blend"], lo[r["item"]], hi[r["item"]])), axis=1)
    return out
