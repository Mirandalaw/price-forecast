"""공공데이터포털 'aT 농수축산물 일자별 도소매 가격'(시장별 KAMIS 조사가격, 연도별 CSV, cp949)
→ data/kamis_market_daily.csv (5품목 × 조사일, 도매 상품, kg당 전국 시장 평균).

사용:  python scripts/build_kamis_market.py <원본 CSV 폴더> [--out data/kamis_market_daily.csv]

규칙
  - 조사구분 = 도매, 등급 = 상품만 쓴다.
  - 품목: 배추, 무, 양파, 깐마늘(국산), 건고추. 품종(작형)은 구분하지 않고 그날 조사된 시장×품종 행을 모두 평균한다
    (경락가 평균에 가까운 정의. 가을배추·고랭지배추가 겹치는 전환기에 급격한 단절을 피한다).
  - kg당 가격 = 품목가격 ÷ 도매출하단위크기 (배추 10kg, 무 18/20kg, 양파 15/20kg, 깐마늘 20kg, 건고추 30/60kg 등 단위 변경을 흡수).
원본은 연간 60~130MB 라 청크로 읽는다. 원본 CSV 는 git 에 올리지 않는다.
"""
import argparse
import glob
import re
from pathlib import Path

import pandas as pd

COLS = ["가격등록일자", "시장명", "품목명", "품종명", "조사구분명", "산물등급명", "품목가격", "도매출하단위크기"]
ITEM_MAP = {"배추": "배추", "무": "무", "양파": "양파", "깐마늘(국산)": "깐마늘", "건고추": "건고추"}
_PRE = re.compile("배추|무|양파|마늘|고추")


def extract(paths):
    out = []
    for f in paths:
        for ch in pd.read_csv(f, encoding="cp949", dtype=str, usecols=lambda c: c in COLS, chunksize=500_000):
            m = ch["품목명"].str.contains(_PRE, na=False) & (ch["조사구분명"].str.strip() == "도매")
            out.append(ch[m])
        print(Path(f).name, flush=True)
    return pd.concat(out, ignore_index=True)


def aggregate(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["산물등급명"].str.strip() == "상품"]
    d = d[d["품목명"].isin(ITEM_MAP)].copy()
    d["item"] = d["품목명"].map(ITEM_MAP)
    d["date"] = pd.to_datetime(d["가격등록일자"], format="%Y%m%d")
    d["price"] = pd.to_numeric(d["품목가격"], errors="coerce")
    d["unit"] = pd.to_numeric(d["도매출하단위크기"], errors="coerce")
    d = d[(d["price"] > 0) & (d["unit"] > 0)]
    d["pkg"] = d["price"] / d["unit"]
    a = (d.groupby(["item", "date"])
          .agg(price=("pkg", "mean"), n_obs=("pkg", "size"),
               kinds=("품종명", lambda s: "|".join(sorted(set(s)))))
          .reset_index().sort_values(["item", "date"]))
    a["price"] = a["price"].round(2)
    return a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="원본 CSV 폴더")
    ap.add_argument("--out", default="data/kamis_market_daily.csv")
    a = ap.parse_args()
    paths = sorted(glob.glob(str(Path(a.src) / "*.csv")))
    if not paths:
        raise SystemExit(f"{a.src} 에 CSV 없음")
    res = aggregate(extract(paths))
    res.to_csv(a.out, index=False)
    print(res.groupby("item")["date"].agg(["min", "max", "count"]))
