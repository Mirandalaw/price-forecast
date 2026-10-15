"""순(旬)별 예측 타깃: 기준일 t 가 속한 순에서 k순 뒤(k=2,3)의 순 평균가격.

순 정의: 상순 1~10일, 중순 11~20일, 하순 21일~말일. 순 평균 = 그 순에 속한 거래일 가격의 산술평균.
(본선 데이터 명세가 다른 집계를 쓰면 sun_mean() 하나만 바꾼다.)

타깃 (features.build 가 호출):
  ys{k} = log(순평균_{sun(t)+k}) − log(P_t)      k ∈ SUN_HORIZONS
  ps{k} = 순평균_{sun(t)+k}
daily 쪽 y{h}/p{h} 와 같은 이름 규칙이라 models.predict_all / features.seasonal_drift 를 그대로 쓴다.
"""
import numpy as np
import pandas as pd

SUN_HORIZONS = ["s2", "s3"]
_K = {"s2": 2, "s3": 3}


def sun_index(d: pd.Series) -> pd.Series:
    """날짜 → 전역 순 번호 (연속 정수). 월 3순 × 12 = 36순/년."""
    part = np.where(d.dt.day <= 10, 0, np.where(d.dt.day <= 20, 1, 2))
    return d.dt.year * 36 + (d.dt.month - 1) * 3 + part


def sun_label(idx: int) -> str:
    y, r = divmod(int(idx), 36)
    m, part = divmod(r, 3)
    return f"{y}-{m + 1:02d} {'상중하'[part]}순"


def sun_mean(g: pd.DataFrame) -> pd.Series:
    """품목 1개의 거래일 가격 → 순 번호별 평균가격."""
    return g.groupby(sun_index(g["date"]))["price"].mean()


def add_targets(g: pd.DataFrame) -> pd.DataFrame:
    """_per_item 결과(품목 1개, date 정렬)에 ys2/ys3, ps2/ps3 열 추가."""
    sm = sun_mean(g)
    si = sun_index(g["date"])
    for name, k in _K.items():
        target = (si + k).map(sm)
        g[f"p{name}"] = target.values
        g[f"y{name}"] = np.log(target.values) - g["logp"].values
    return g


def target_label(asof: pd.Timestamp, name: str) -> str:
    return sun_label(int(sun_index(pd.Series([asof])).iloc[0]) + _K[name])
