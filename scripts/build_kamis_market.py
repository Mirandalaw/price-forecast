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

import numpy as np
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
    """5품목 × 조사일: 상품 kg당 시장 평균(price) + 시장 구조 지표.
    mk_qspread  = log(상품 평균 / 중품 평균)            품질 스프레드(상품만 귀하면 커진다 → 공급 부족 신호)
    mk_garak    = log(가락 상품 / 가락 외 시장 상품 평균)  수도권-지방 가격 격차
    mk_disp     = 시장별 상품 로그가격 표준편차             시장 간 분산(교란 신호)
    mk_nkinds   = 그날 조사된 품종(작형) 수
    mk_kindage  = 조사 품종 구성이 마지막으로 바뀐 뒤 지난 조사일 수(최대 30) — 가을배추 첫 등장 등 작형 전환 신호
    """
    d = d[d["품목명"].isin(ITEM_MAP)].copy()
    d["item"] = d["품목명"].map(ITEM_MAP)
    d["grade"] = d["산물등급명"].str.strip()
    d = d[d["grade"].isin(["상품", "중품"])]
    d["date"] = pd.to_datetime(d["가격등록일자"], format="%Y%m%d")
    d["price"] = pd.to_numeric(d["품목가격"], errors="coerce")
    d["unit"] = pd.to_numeric(d["도매출하단위크기"], errors="coerce")
    d = d[(d["price"] > 0) & (d["unit"] > 0)]
    d["pkg"] = d["price"] / d["unit"]
    d["lpkg"] = np.log(d["pkg"])
    hi = d[d["grade"] == "상품"]
    mid = d[d["grade"] == "중품"]
    a = (hi.groupby(["item", "date"])
           .agg(price=("pkg", "mean"), n_obs=("pkg", "size"),
                kinds=("품종명", lambda s: "|".join(sorted(set(s)))),
                mk_disp=("lpkg", "std"), mk_nkinds=("품종명", "nunique"))
           .reset_index())
    m = mid.groupby(["item", "date"])["pkg"].mean().rename("p_mid").reset_index()
    a = a.merge(m, on=["item", "date"], how="left")
    a["mk_qspread"] = np.log(a["price"] / a["p_mid"])
    g = hi[hi["시장명"].str.contains("가락")].groupby(["item", "date"])["pkg"].mean().rename("p_garak")
    o = hi[~hi["시장명"].str.contains("가락")].groupby(["item", "date"])["pkg"].mean().rename("p_other")
    a = a.merge(g.reset_index(), on=["item", "date"], how="left").merge(o.reset_index(), on=["item", "date"], how="left")
    a["mk_garak"] = np.log(a["p_garak"] / a["p_other"])
    a = a.sort_values(["item", "date"]).reset_index(drop=True)
    changed = a.groupby("item")["kinds"].transform(lambda s: s.ne(s.shift()))
    age = []
    for _, grp in a.groupby("item", sort=False):
        k = 30
        for c in changed.loc[grp.index]:
            k = 0 if c else min(k + 1, 30)
            age.append(k)
    a["mk_kindage"] = age
    a["mk_disp"] = a["mk_disp"].fillna(0.0)
    for c in ["price", "mk_qspread", "mk_garak", "mk_disp"]:
        a[c] = a[c].round(4)
    a["price"] = a["price"].round(2)
    cols = ["item", "date", "price", "n_obs", "kinds", "mk_qspread", "mk_garak", "mk_disp", "mk_nkinds", "mk_kindage"]
    return a[cols]


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
