"""워크포워드 백테스트: 연도별로 (학습 = 그 해 10/25 이전 전체, 평가 = 그 해 10/26~11/09 기준일) × h=1..10."""
import numpy as np
import pandas as pd
from . import features as F, models as M


def run(prices: pd.DataFrame, years=(2021, 2022, 2023, 2024), horizons=F.HORIZONS,
        start="10-26", end="11-09") -> pd.DataFrame:
    feat = F.build(prices)
    rows = []
    for h in horizons:
        feat["drift"] = F.seasonal_drift(feat, h)
        for yr in years:
            cut = pd.Timestamp(f"{yr}-{start}")
            train = feat[feat["date"] < cut - pd.Timedelta(days=h)]      # 타깃이 평가 구간을 넘보지 않게
            test = feat[(feat["date"] >= cut) & (feat["date"] <= pd.Timestamp(f"{yr}-{end}"))].dropna(subset=[f"p{h}"])
            if test.empty:
                continue
            pr = M.predict_all(train, test, h)
            for mdl in ("persist", "drift_m", "lgbm", "blend"):
                pred_price = pr["price"] * np.exp(pr[mdl])
                ape = (pred_price - pr[f"p{h}"]).abs() / pr[f"p{h}"]
                for item, a in ape.groupby(pr["item"]):
                    rows.append(dict(year=yr, h=h, model=mdl, item=item, mape=a.mean() * 100, n=len(a)))
    return pd.DataFrame(rows)


def report(res: pd.DataFrame) -> str:
    lines = []
    for h in sorted(res["h"].unique()):
        if h not in (1, 3, 5, 10):
            continue
        t = res[res["h"] == h].groupby(["item", "model"])["mape"].mean().unstack("model").round(1)
        t = t[["persist", "drift_m", "lgbm", "blend"]]
        t.loc["5품목 평균"] = t.mean().round(1)
        lines.append(f"\n== h={h} 거래일 앞 · MAPE(%) · 2021~2024 평균 ==\n{t.to_string()}")
    return "\n".join(lines)
