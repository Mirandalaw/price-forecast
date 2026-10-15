"""피처 생성. 모두 '예측 기준일 t까지의 값'만 쓴다 (미래 정보 금지).

seasonal_drift: 같은 달력 구간에서 과거 연도들의 h일 뒤 로그변화율 평균 (해당 연도 제외 = leave-one-year-out).
"""
import numpy as np
import pandas as pd

HORIZONS = list(range(1, 11))


def _per_item(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    p = g["price"]
    g["logp"] = np.log(p)
    g["base7"] = p.rolling(7).median()
    g["ret1"] = g["logp"].diff()
    for k in (1, 2, 3, 5, 7, 14):
        g[f"lag_ret{k}"] = g["logp"].diff(k)
    g["ma7_gap"] = np.log(p / p.rolling(7).mean())
    g["ma28_gap"] = np.log(p / p.rolling(28).mean())
    g["vol14"] = g["ret1"].rolling(14).std()
    g["dow"] = g["date"].dt.dayofweek
    g["doy"] = g["date"].dt.dayofyear
    g["year"] = g["date"].dt.year
    g["sun_gap"] = np.log(p / g["p_sun"])            # 당일가 vs 이번 순 평균
    g["normal_gap"] = np.log(g["p_sun"] / g["p_normal_sun"])  # 평년 대비 수준
    g["ly_gap"] = np.log(g["p_sun"] / g["p_ly_sun"])
    for h in HORIZONS:                                # 타깃: h거래일 뒤 로그변화율
        g[f"y{h}"] = g["logp"].shift(-h) - g["logp"]
        g[f"p{h}"] = p.shift(-h)
    return g


def build(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([_per_item(g) for _, g in prices.groupby("item")], ignore_index=True)


def seasonal_drift(feat: pd.DataFrame, h: int, window: int = 7, min_n: int = 5) -> pd.Series:
    """각 행에 대해: 다른 연도들에서 doy±window 안의 y{h} 평균 (leave-one-year-out).

    연도×일(1..366) 격자에 y를 놓고 window 합을 구한 뒤, 전체 합에서 자기 연도분을 뺀다.
    """
    out = pd.Series(np.nan, index=feat.index)
    k = np.ones(2 * window + 1)
    for item, g in feat.groupby("item"):
        years = np.sort(g["year"].unique()); yi = {y: i for i, y in enumerate(years)}
        Y = np.full((len(years), 367), np.nan)
        Y[g["year"].map(yi).values, g["doy"].values] = g[f"y{h}"].values
        V = np.nan_to_num(Y); C = (~np.isnan(Y)).astype(float)
        S = np.vstack([np.convolve(r, k, mode="same") for r in V]); N = np.vstack([np.convolve(r, k, mode="same") for r in C])
        tot_s, tot_n = S.sum(0), N.sum(0)
        rows = g["year"].map(yi).values; cols = g["doy"].values
        s = tot_s[cols] - S[rows, cols]; n = tot_n[cols] - N[rows, cols]
        out[g.index] = np.where(n >= min_n, s / np.maximum(n, 1), np.nan)
    return out
