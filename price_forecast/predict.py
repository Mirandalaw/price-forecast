"""기준일(asof) 하나에 대한 제출용 예측: 5품목 × (1~10거래일 후, 2순·3순 후).

규칙
  - asof 이하의 가격(및 뉴스)만 쓴다. asof 가 거래일이 아니면 직전 거래일 행을 기준 행으로 쓴다.
  - 예측일(h)마다 모델을 따로 학습한다 (backtest 와 같은 코드 경로 → 백테스트 수치가 곧 기대 성능).
  - 어떤 이유로든 모델이 실패하면 그 (품목, h) 는 persistence(기준가 유지)로 떨어진다. 제출 파일은 항상 완성된다.

출력 열: asof, item, horizon, target, base_price, pred_price, yhat, model
  horizon : 1..10 또는 s2/s3
  target  : 일별은 거래일 h일 뒤의 '추정 달력일'(월~토 영업일 가정, 본선 명세로 교체), 순별은 '2026-11 중순' 식 순 라벨
"""
import logging

import numpy as np
import pandas as pd

from . import features as F, models as M, news as N, sun, weather as W

log = logging.getLogger(__name__)
DEFAULT_MODEL = "blend"
# model="auto": 품목별 모델 (README '품목별 운용'. 2017~2024 8개년 시장별 자료 + 2021~2024 at5 자료 백테스트 근거)
#   배추·무 = 뉴스 피처 LightGBM (10일 앞 8개년: 배추 16.8→15.0%, 무 14.2→12.4%) / 저장품 = 기준가 유지 (모델이 오히려 악화)
#   뉴스 신호가 없으면 배추·무는 가격 피처만의 lgbm 으로 내려간다. (날씨 lgbm_wx 는 drift 누수 제거 후 이득이 사라져 제외)
ITEM_MODEL = {"배추": "lgbm_news", "무": "lgbm_news", "양파": "persist", "깐마늘": "persist", "건고추": "persist"}


def _target_date(asof: pd.Timestamp, h: int) -> pd.Timestamp:
    """월~토 영업일 기준 h일 뒤 (공휴일 미반영 — 본선 거래일 명세가 오면 교체)."""
    d = asof
    n = 0
    while n < h:
        d += pd.Timedelta(days=1)
        if d.dayofweek != 6:
            n += 1
    return d


def forecast(prices: pd.DataFrame, asof: str | pd.Timestamp, signals: pd.DataFrame | None = None,
             model: str = DEFAULT_MODEL, horizons=F.ALL_HORIZONS, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    asof = pd.Timestamp(asof)
    hist = prices[prices["date"] <= asof]
    if hist.empty:
        raise ValueError(f"{asof.date()} 이전 가격 자료가 없음")
    wx = weather[weather["date"] <= asof] if weather is not None else None
    feat = W.attach(N.attach(F.build(hist), signals), wx)
    has_news = signals is not None and not signals.empty
    use_news = has_news and model.endswith("_news")
    has_wx = wx is not None and not wx.empty
    if model == "auto":
        def _fallback(m):
            if m == "lgbm_wx" and not has_wx:
                return "blend"
            if m == "lgbm_news" and not has_news:
                return "lgbm"
            return m
        per_item = {i: _fallback(m) for i, m in ITEM_MODEL.items()}
    else:
        per_item = None
    base_model = model.replace("_news", "")
    # 기준 행 = 품목별 마지막 거래일
    last_idx = feat.sort_values("date").groupby("item").tail(1).index
    last = feat.loc[last_idx].set_index("item")
    rows = []
    for h in horizons:
        feat["drift"] = F.seasonal_drift(feat, h)
        cur = feat.loc[last_idx]
        train = feat.dropna(subset=[f"y{h}"])
        try:
            if per_item is not None:
                pr = M.predict_all(train, cur, h).set_index("item")
                need = set(per_item.values())
                prw = M.predict_all(train, cur, h, use_wx=True).set_index("item") if "lgbm_wx" in need else pr
                prn = M.predict_all(train, cur, h, use_news=True).set_index("item") if "lgbm_news" in need else pr
                src = {"lgbm_wx": (prw, "lgbm"), "lgbm_news": (prn, "lgbm")}
                pick = {}
                for i, m in per_item.items():
                    if i not in pr.index:
                        continue
                    tab, col = src.get(m, (pr, m))
                    pick[i] = tab[col][i]
                yhat = pd.Series(pick)
                used = {i: f"auto:{per_item[i]}" for i in yhat.index}
            else:
                pr = M.predict_all(train, cur, h, use_news=use_news)
                yhat = pr.set_index("item")[base_model]
                used = {i: model for i in yhat.index}
        except Exception as e:  # noqa: BLE001 — 제출 파일은 반드시 나와야 한다
            log.warning("h=%s 모델 실패 → persistence 로 대체: %s", h, e)
            yhat = pd.Series(0.0, index=last.index)
            used = {i: "persist(fallback)" for i in yhat.index}
        for item in last.index:
            y = float(yhat.get(item, 0.0))
            if not np.isfinite(y):
                y, used[item] = 0.0, "persist(fallback)"
            base = float(last.loc[item, "price"])
            target = sun.target_label(asof, h) if isinstance(h, str) else _target_date(asof, h).date().isoformat()
            rows.append(dict(asof=asof.date().isoformat(), item=item, horizon=h, target=target,
                             base_price=round(base, 1), pred_price=round(base * float(np.exp(y)), 1),
                             yhat=round(y, 5), model=used[item]))
    out = pd.DataFrame(rows)
    order = {h: i for i, h in enumerate(horizons)}
    return out.sort_values(["item", "horizon"], key=lambda s: s.map(order) if s.name == "horizon" else s).reset_index(drop=True)
