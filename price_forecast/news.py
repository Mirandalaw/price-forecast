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


# ---------------------------------------------------------------------------
# 이벤트 스터디 보정 (기획서 4-5 / 기술서 1-1 ②): 사건 유형별 "보도 후 h일 초과 반응"을 과거에서 재고,
# 기준일 직전 k일 안에 그 사건이 있으면 기본 예측값에 더한다. 학습 연도 자료로만 추정한다.
# ---------------------------------------------------------------------------
EVT_WINDOW_DAYS = 3        # 기준일 직전 며칠 안의 사건을 보정에 쓸지
EVT_SHRINK = 0.5           # 추정 반응의 몇 %를 쓸지 (과적합 방지)
EVT_MIN_N = 10             # 유형별 최소 사건 수
EVT_MIN_HIT = 0.58         # 학습 구간에서 방향 적중률이 이 미만인 유형은 보정에 쓰지 않음 (기획서 4-5)
EVT_ITEMS = ("배추", "무")  # 뉴스 보정을 적용할 품목 (저장품은 사건 반응이 작고 표본이 적어 제외)
_H_DAYS = {"s2": 20, "s3": 30}  # 순별 지평을 거래일 수로 근사


def event_table(signals: pd.DataFrame) -> pd.DataFrame:
    """signals(일자·품목 집계) → 사건 1건당 1행 (date, item, rule)."""
    ev = [c for c in signals.columns if c.startswith("evt_")]
    rows = []
    for _, r in signals.iterrows():
        for c in ev:
            for _ in range(int(r[c])):
                rows.append(dict(date=r["date"], item=r["item"], rule=c[4:]))
    return pd.DataFrame(rows, columns=["date", "item", "rule"])


def event_study(prices: pd.DataFrame, signals: pd.DataFrame, before: pd.Timestamp, horizons) -> pd.DataFrame:
    """사건 유형별 초과 반응: E[logP(t+h) − logP(t) | 사건] − E[같은 품목 8~11월 무조건].

    before 이전 날짜의 사건·가격만 쓴다 (워크포워드에서 미래 정보 차단). 품목을 합쳐 유형별로 추정.
    반환: index=rule, columns=h → 초과 반응 (n < EVT_MIN_N 이면 0), 'n' 열 포함.
    """
    E = event_table(signals)
    E = E[(E["date"] < before) & (E["item"].isin(EVT_ITEMS))]
    hs = {h: (_H_DAYS.get(h, h) if isinstance(h, str) else h) for h in horizons}
    recs = []
    for item, g in prices[(prices["date"] < before) & (prices["item"].isin(EVT_ITEMS))].groupby("item"):
        lp = np.log(g.set_index("date")["price"])
        aut = np.where(lp.index.month.isin([8, 9, 10, 11]))[0]
        base = {h: np.nanmean([lp.iloc[p + d] - lp.iloc[p] for p in aut if p + d < len(lp)]) if len(aut) else 0.0
                for h, d in hs.items()}
        for _, e in E[E["item"] == item].iterrows():
            pos = lp.index.searchsorted(e["date"])
            rec = dict(item=item, rule=e["rule"])
            for h, d in hs.items():
                rec[h] = (lp.iloc[pos + d] - lp.iloc[pos] - base[h]) if pos + d < len(lp) else np.nan
            recs.append(rec)
    if not recs:
        return pd.DataFrame(columns=["item"] + list(hs) + ["n"])
    R = pd.DataFrame(recs)
    tab = R.groupby(["item", "rule"]).mean(numeric_only=True)
    tab["n"] = R.groupby(["item", "rule"]).size()
    for h in hs:  # 방향 적중률: 초과 반응의 부호가 유형 평균 부호와 같은 비율
        hits = R.groupby(["item", "rule"])[h].apply(lambda v: np.mean(np.sign(v.dropna()) == np.sign(v.dropna().mean())) if len(v.dropna()) else 0.0)
        weak = (tab["n"] < EVT_MIN_N) | (hits < EVT_MIN_HIT)
        tab.loc[weak, h] = 0.0
    return tab.fillna(0.0)


def event_adjustment(test: pd.DataFrame, signals: pd.DataFrame, table: pd.DataFrame, h) -> pd.Series:
    """test 행(item, date)마다: 기준일 직전 EVT_WINDOW_DAYS 안의 사건 유형별 초과 반응 합 × EVT_SHRINK."""
    out = pd.Series(0.0, index=test.index)
    if table.empty or h not in table.columns:
        return out
    E = event_table(signals)
    for i, r in test[["item", "date"]].iterrows():
        if r["item"] not in EVT_ITEMS:
            continue
        w = E[(E["item"] == r["item"]) & (E["date"] <= r["date"]) & (E["date"] > r["date"] - pd.Timedelta(days=EVT_WINDOW_DAYS))]
        if w.empty:
            continue
        # 같은 유형이 여러 건이어도 1건으로 센다 (매체 수 중복 방지)
        out[i] = EVT_SHRINK * sum(table[h].get((r["item"], rule), 0.0) for rule in w["rule"].unique())
    return out
