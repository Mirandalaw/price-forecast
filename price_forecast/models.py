"""예측 모델 3종. 모두 '로그변화율 ŷ_h'를 내고, 가격은 P(t)·exp(ŷ_h).

persistence : ŷ = 0
drift       : ŷ = seasonal_drift (과거 연도 같은 시기 평균 변화율)
lgbm        : 예측일별 LightGBM 10개, L1 목적함수 + 1/실제가격 가중(MAPE 정합), 피처 = 래그·MA·평년비·drift
blend       : lgbm·drift 가중 평균 후 h가 멀수록 drift 쪽으로 수축, 품목별 과거 1%·99% 분위로 클리핑
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

FEATS = ["lag_ret1", "lag_ret2", "lag_ret3", "lag_ret5", "lag_ret7", "lag_ret14",
         "ma7_gap", "ma28_gap", "vol14", "dow", "doy", "sun_gap", "normal_gap", "ly_gap", "drift"]
PARAMS = dict(objective="l1", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)


def fit_lgbm(train: pd.DataFrame, h: int) -> lgb.Booster:
    t = train.dropna(subset=FEATS + [f"y{h}", f"p{h}"])
    w = 1.0 / t[f"p{h}"].clip(lower=1)          # MAPE 가중: 실제가격이 낮을수록 오차가 비싸다
    w = w / w.mean()
    ds = lgb.Dataset(t[FEATS], t[f"y{h}"], weight=w, categorical_feature=["dow"])
    return lgb.train(PARAMS, ds, num_boost_round=400)


def clip_by_item(yhat: pd.Series, train: pd.DataFrame, h: int) -> pd.Series:
    lo = train.groupby("item")[f"y{h}"].quantile(0.01); hi = train.groupby("item")[f"y{h}"].quantile(0.99)
    items = yhat.index.get_level_values("item") if isinstance(yhat.index, pd.MultiIndex) else None
    return yhat  # 호출부에서 품목별로 적용


def predict_all(train: pd.DataFrame, test: pd.DataFrame, h: int, w_lgbm: float = 0.6) -> pd.DataFrame:
    """test 행마다 persistence / drift / lgbm / blend 의 ŷ_h 를 돌려준다."""
    out = test[["item", "date", "price", f"p{h}", f"y{h}", "drift"]].copy()
    out["persist"] = 0.0
    out["drift_m"] = out["drift"].fillna(0.0)
    m = fit_lgbm(train, h)
    x = test[FEATS].copy()
    out["lgbm"] = np.where(x.isna().any(axis=1), out["drift_m"], m.predict(x.fillna(0)))
    shrink = 1 - 0.04 * (h - 1)                  # h=1 → 1.0, h=10 → 0.64 : 멀수록 drift로 수축
    out["blend"] = (w_lgbm * shrink) * out["lgbm"] + (1 - w_lgbm * shrink) * out["drift_m"]
    lo = train.groupby("item")[f"y{h}"].quantile(0.01); hi = train.groupby("item")[f"y{h}"].quantile(0.99)
    out["blend"] = out.apply(lambda r: float(np.clip(r["blend"], lo[r["item"]], hi[r["item"]])), axis=1)
    return out
