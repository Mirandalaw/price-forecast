"""데이터 로더.

지금은 공공데이터포털 '주요 채소류 일일가격 현황'(aT, KAMIS 도매 조사가격) CSV를 읽는다.
본선 표준 데이터셋(경락가·반입량)이 오면 load_auction() 하나만 추가하고 load_prices()의 반환 형식을
맞추면 나머지 코드는 그대로 돈다.

반환 형식 (long):  item, date, price  — item ∈ {배추, 무, 양파, 깐마늘, 건고추}, date = 가격 관측일(거래일)
"""
from pathlib import Path
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


def load_prices(data_dir: str | Path = "data") -> pd.DataFrame:
    """data/ 아래 aT CSV 전부를 읽는다. 본선 데이터셋이 생기면 여기서 분기."""
    paths = sorted(Path(data_dir).glob("*.csv"))
    paths = [p for p in paths if "일일가격" in p.name or "at5_raw" in p.name]
    if not paths:
        raise FileNotFoundError(f"{data_dir} 에 aT 일일가격 CSV 없음")
    return load_at5(paths)
