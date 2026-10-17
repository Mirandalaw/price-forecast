"""워크포워드 백테스트: 연도별로 (학습 = 그 해 10/25 이전 전체, 평가 = 그 해 10/26~11/09 기준일) × h=1..10."""
import numpy as np
import pandas as pd
from . import features as F, models as M, news as N, weather as W


K_GRID = np.round(np.arange(0.90, 1.021, 0.01), 2)


def mape_shrink_factor(pairs: list[tuple[float, float]]) -> float:
    """(예측가, 실제가) 쌍들에서 MAPE 를 최소화하는 곱셈 계수 k ∈ K_GRID. 쌍이 없으면 1.0.

    MAPE 는 실제가가 낮을 때 오차가 비싸므로 최적 k 는 보통 1보다 약간 작다 (기술서 1-1 ③ 안전장치의 한 형태).
    """
    if not pairs:
        return 1.0
    p = np.array([a for a, _ in pairs]); y = np.array([b for _, b in pairs])
    return float(K_GRID[np.argmin([np.mean(np.abs(k * p - y) / y) for k in K_GRID])])


def run(prices: pd.DataFrame, years=(2021, 2022, 2023, 2024), horizons=F.ALL_HORIZONS,
        start="10-26", end="11-09", signals: pd.DataFrame | None = None,
        weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """signals 를 주면 'lgbm_news' / 'blend_news' / 'blend_evt' 모델이 추가된다 (뉴스 on/off 비교).

    'blend_k' = blend × 품목·h별 MAPE 수축 계수. 계수는 **앞선 평가 연도들의 out-of-sample 예측**으로만 고른다
    (첫 해는 1.0). 연도 순서대로 돌기 때문에 미래 정보가 섞이지 않는다.
    """
    feat = W.attach(N.attach(F.build(prices), signals), weather)
    use_news = signals is not None and not signals.empty
    use_wx = weather is not None and not weather.empty
    rows = []
    for h in horizons:
        feat["drift"] = F.seasonal_drift(feat, h)
        history: dict[str, list[tuple[float, float]]] = {}   # item → [(blend 예측가, 실제가)] 앞선 연도분
        for yr in sorted(years):
            cut = pd.Timestamp(f"{yr}-{start}")
            gap = pd.Timedelta(days=40 if isinstance(h, str) else h)      # 타깃이 평가 구간을 넘보지 않게 (순별은 3순 ≈ 30일 + 여유)
            train = feat[feat["date"] < cut - gap]
            test = feat[(feat["date"] >= cut) & (feat["date"] <= pd.Timestamp(f"{yr}-{end}"))].dropna(subset=[f"p{h}"])
            if test.empty:
                continue
            pr = M.predict_all(train, test, h)
            if use_news:
                pn = M.predict_all(train, test, h, use_news=True)
                pr["lgbm_news"], pr["blend_news"] = pn["lgbm"], pn["blend"]
                tab = N.event_study(prices, signals, cut - gap, [h])
                adj = N.event_adjustment(test, signals, tab, h)
                lo = train.groupby("item")[f"y{h}"].quantile(0.01); hi = train.groupby("item")[f"y{h}"].quantile(0.99)
                pr["blend_evt"] = [float(np.clip(b + a, lo[it], hi[it])) for b, a, it in zip(pr["blend"], adj, pr["item"])]
            if use_wx:
                pw = M.predict_all(train, test, h, use_wx=True)
                pr["lgbm_wx"], pr["blend_wx"] = pw["lgbm"], pw["blend"]
            pr["k"] = pr["item"].map(lambda it: mape_shrink_factor(history.get(it, [])))
            pr["blend_k"] = pr["blend"] + np.log(pr["k"])
            for it, g in pr.groupby("item"):
                history.setdefault(it, []).extend(zip(g["price"] * np.exp(g["blend"]), g[f"p{h}"]))
            for mdl in [c for c in CANDIDATES if c in pr]:
                pred_price = pr["price"] * np.exp(pr[mdl])
                ape = (pred_price - pr[f"p{h}"]).abs() / pr[f"p{h}"]
                for item, a in ape.groupby(pr["item"]):
                    rows.append(dict(year=yr, h=h, model=mdl, item=item, mape=a.mean() * 100, n=len(a)))
    return pd.DataFrame(rows)


CANDIDATES = ("persist", "drift_m", "lgbm", "blend", "blend_k", "lgbm_news", "blend_news", "blend_evt", "lgbm_wx", "blend_wx")


def add_selection(res: pd.DataFrame, min_years: int = 1) -> pd.DataFrame:
    """품목·예측일별 모델 선택을 '그 이전 연도 성적'으로만 한다 (미래 정보 없음).

    평가 연도 Y 의 (item, h) 에는 Y 보다 앞선 연도들의 평균 MAPE 가 가장 낮은 모델을 쓴다.
    앞선 연도가 min_years 개 미만이면 persist. 결과는 model="select" 행으로 추가된다.
    기술서 1-1 '품목별로 다르게 다룹니다' 의 구현: 저장품(깐마늘·건고추)은 보통 persist, 배추는 lgbm 쪽으로 간다.
    """
    rows = []
    for (h, item), g in res.groupby(["h", "item"], sort=False):
        for yr in sorted(g["year"].unique()):
            past = g[g["year"] < yr]
            if past["year"].nunique() < min_years:
                pick = "persist"
            else:
                pick = past.groupby("model")["mape"].mean().idxmin()
            r = g[(g["year"] == yr) & (g["model"] == pick)].iloc[0]
            rows.append(dict(year=yr, h=h, model="select", item=item, mape=r["mape"], n=r["n"], picked=pick))
    return pd.concat([res, pd.DataFrame(rows)], ignore_index=True)


def report(res: pd.DataFrame) -> str:
    if "select" not in set(res["model"]):
        res = add_selection(res)
    lines = []
    hs = res["h"].astype(str)
    for h in [h for h in ("1", "3", "5", "10", "s2", "s3") if h in set(hs)]:
        t = res[hs == h].groupby(["item", "model"])["mape"].mean().unstack("model").round(1)
        t = t[[c for c in CANDIDATES + ("select",) if c in t.columns]]
        t.loc["5품목 평균"] = t.mean().round(1)
        label = {"s2": "2순 후 순평균", "s3": "3순 후 순평균"}.get(h, f"h={h} 거래일 앞")  # h 는 str
        lines.append(f"\n== {label} · MAPE(%) · 연도 평균 ==\n{t.to_string()}")
    return "\n".join(lines)
