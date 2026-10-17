"""데이터 로더.

지금은 공공데이터포털 '주요 채소류 일일가격 현황'(aT, KAMIS 도매 조사가격) CSV를 읽는다.
본선 표준 데이터셋(경락가·반입량)이 오면 load_auction() 하나만 추가하고 load_prices()의 반환 형식을
맞추면 나머지 코드는 그대로 돈다.

반환 형식 (long):  item, date, price  — item ∈ {배추, 무, 양파, 깐마늘, 건고추}, date = 가격 관측일(거래일)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ITEMS = ["배추", "무", "양파", "깐마늘", "건고추"]
_RENAME = {"마늘": "깐마늘"}

# 2023 파일의 마늘 품종코드: 12A9 = 깐마늘(국산) 으로 가정 (2024 파일의 마늘 가격 수준과 일치), 1209 = 제외
_GARLIC_KEEP = "12A9"


def load_at5(paths: list[str | Path]) -> pd.DataFrame:
    """aT 주요 채소류 일일가격 현황 CSV 여러 개 → 거래일 기준 long 테이블."""
    frames = []
    for p in paths:
        d = pd.read_csv(p, encoding="cp949", dtype=str)
        if "품종코드" in d.columns:
            d = d[~((d["품목명"] == "마늘") & (d["품종코드"] != _GARLIC_KEEP))]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["item"] = df["품목명"].replace(_RENAME)
    df = df[df["item"].isin(ITEMS)].copy()
    df["qdate"] = pd.to_datetime(df["조회일자"])
    df["date"] = pd.to_datetime(df["도매일자"], errors="coerce")
    for c in ["도매일일가격", "도매순가격", "도매전순가격", "도매전년순가격", "도매평년순가격"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["item", "qdate"]).drop_duplicates(["item", "qdate"])
    # 거래일 = 도매일자가 바뀐 날 (휴장일엔 직전 가격이 반복 조회됨)
    df = df[df.groupby("item")["date"].transform(lambda s: s.ne(s.shift()))]
    out = df.rename(columns={"도매일일가격": "price", "도매순가격": "p_sun", "도매전순가격": "p_prev_sun",
                             "도매전년순가격": "p_ly_sun", "도매평년순가격": "p_normal_sun"})
    cols = ["item", "date", "price", "p_sun", "p_prev_sun", "p_ly_sun", "p_normal_sun"]
    out = out[cols].dropna(subset=["price"]).drop_duplicates(["item", "date"]).sort_values(["item", "date"])
    return out.reset_index(drop=True)


def _sun_columns(g: pd.DataFrame) -> pd.DataFrame:
    """일별 시리즈(품목 1개)에서 KAMIS 식 순 지표를 만든다.
    p_sun = 이번 순의 현재까지 평균, p_prev_sun = 직전 순 평균, p_ly_sun = 작년 같은 순 평균,
    p_normal_sun = 평년(직전 5개년 같은 순 평균 중 최대·최소 제외 3개년 평균; 5년 미만이면 있는 만큼 평균, 2년 미만이면 NaN)."""
    from . import sun
    g = g.sort_values("date").copy()
    si = sun.sun_index(g["date"])
    g["p_sun"] = g.groupby(si)["price"].transform(lambda s: s.expanding().mean())
    full = g.groupby(si)["price"].mean()
    g["p_prev_sun"] = (si - 1).map(full)
    g["p_ly_sun"] = (si - 36).map(full)

    def normal(i):
        v = [full.get(i - 36 * k) for k in range(1, 6)]
        v = sorted(x for x in v if x is not None and np.isfinite(x))
        if len(v) < 2:
            return np.nan
        if len(v) == 5:
            v = v[1:-1]
        return float(np.mean(v))
    g["p_normal_sun"] = si.map({i: normal(i) for i in si.unique()})
    # 직전 순 값이 없으면(자료 시작 직후) 당일 순으로 대체
    g["p_prev_sun"] = g["p_prev_sun"].fillna(g["p_sun"])
    return g


def load_kamis_market(path: str | Path = "data/kamis_market_daily.csv") -> pd.DataFrame:
    """scripts/build_kamis_market.py 출력(시장별 KAMIS 도매 상품 kg당 평균, 2015~) → load_at5 와 같은 형식."""
    d = pd.read_csv(path, parse_dates=["date"])
    d = d[d["item"].isin(ITEMS) & (d["price"] > 0)]
    out = pd.concat([_sun_columns(g) for _, g in d.groupby("item")], ignore_index=True)
    cols = ["item", "date", "price", "p_sun", "p_prev_sun", "p_ly_sun", "p_normal_sun"]
    cols += [c for c in out.columns if c.startswith("mk_")]          # 시장 구조 지표 (market.py)
    return out[cols].sort_values(["item", "date"]).reset_index(drop=True)


def load_prices(data_dir: str | Path = "data", source: str = "at5") -> pd.DataFrame:
    """source="at5": 주요 채소류 일일가격 현황 CSV (2019-10~2024, 기본값 — 기술서 수치의 근거).
    source="market": 시장별 일자별 도소매 가격 집계 (2015~2025-08, kamis_market_daily.csv).
    본선 데이터셋이 생기면 source="auction" 분기 추가."""
    if source == "market":
        return load_kamis_market(Path(data_dir) / "kamis_market_daily.csv")
    paths = sorted(Path(data_dir).glob("*.csv"))
    paths = [p for p in paths if "일일가격" in p.name or "at5_raw" in p.name]
    if not paths:
        raise FileNotFoundError(f"{data_dir} 에 aT 일일가격 CSV 없음")
    return load_at5(paths)
