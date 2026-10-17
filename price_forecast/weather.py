"""주산지 날씨 피처 (기상청 ASOS 시간자료 → 일자료 → 품목별 롤링 피처). 기술서 R04.

입력: data/raw_weather/OBS_ASOS_TIM_*.csv (cp949, 지점·지점명·일시·기온·강수량·풍속·일조·적설·지면온도 …)
     또는 이미 일자료로 만든 data/weather_asos_daily.csv
품목 ↔ 지점 (주산지 근사, 보유 지점만):
  배추·무  : 대관령(100, 고랭지) + 해남(261, 가을·월동)
  양파     : 해남(261)  — 무안(165) 미보유, 같은 전남 서남부로 대체
  깐마늘   : 의성(278) + 해남(261)
  건고추   : 의성(278)  — 영양(801) 미보유, 인접 경북 북부로 대체
피처(기준일 t 포함 과거만): wx_t7(7일 평균기온), wx_tmin7(7일 최저기온), wx_rain7·wx_rain14(누적 강수),
  wx_sun7(7일 일조), wx_frost7(지면 최저 ≤ 0℃ 일수), wx_wind7(7일 최대 풍속), wx_hot7(일 최고 ≥ 30℃ 일수)
날씨 자료가 없는 기간은 NaN 으로 둔다 (LightGBM 이 결측을 직접 처리; 0 으로 채우면 '추웠다'로 오해).
"""
from pathlib import Path

import numpy as np
import pandas as pd

STATIONS = {"배추": [100, 261], "무": [100, 261], "양파": [261], "깐마늘": [278, 261], "건고추": [278]}
WX_FEATS = ["wx_t7", "wx_tmin7", "wx_rain7", "wx_rain14", "wx_sun7", "wx_frost7", "wx_wind7", "wx_hot7"]


def daily_from_hourly(paths) -> pd.DataFrame:
    frames = [pd.read_csv(p, encoding="cp949") for p in paths]
    h = pd.concat(frames, ignore_index=True).drop_duplicates(["지점", "일시"])
    h["ts"] = pd.to_datetime(h["일시"])
    # 'YYYY-MM-DD 00:00' 은 전날 24시 관측 → 전날로 귀속 (기상청 일자료 관례)
    h["date"] = (h["ts"] - pd.Timedelta(minutes=1)).dt.normalize()
    g = h.groupby(["지점", "date"])
    d = pd.DataFrame({
        "tavg": g["기온(°C)"].mean(), "tmin": g["기온(°C)"].min(), "tmax": g["기온(°C)"].max(),
        "rain": g["강수량(mm)"].sum(min_count=1).fillna(0.0), "sun": g["일조(hr)"].sum(min_count=1).fillna(0.0),
        "wind": g["풍속(m/s)"].max(), "gmin": g["지면온도(°C)"].min(), "n": g["기온(°C)"].count(),
    }).reset_index().rename(columns={"지점": "stn"})
    return d[d["n"] >= 18].drop(columns="n")  # 하루 관측이 18시간 미만이면 버림


def load_daily(data_dir="data") -> pd.DataFrame | None:
    data_dir = Path(data_dir)
    cached = data_dir / "weather_asos_daily.csv"
    raws = sorted((data_dir / "raw_weather").glob("*ASOS*.csv"))
    if raws:
        d = daily_from_hourly(raws)
        d.to_csv(cached, index=False)
        return d
    if cached.exists():
        return pd.read_csv(cached, parse_dates=["date"])
    return None


def item_features(daily: pd.DataFrame, items) -> pd.DataFrame:
    out = []
    for item in items:
        stns = STATIONS.get(item, [])
        s = daily[daily["stn"].isin(stns)].groupby("date").mean(numeric_only=True).drop(columns="stn", errors="ignore")
        if s.empty:
            continue
        s = s.asfreq("D")  # 결측일은 NaN 유지
        f = pd.DataFrame(index=s.index)
        r = lambda col, w, fn: getattr(s[col].rolling(w, min_periods=max(1, w // 2)), fn)()  # noqa: E731
        f["wx_t7"] = r("tavg", 7, "mean")
        f["wx_tmin7"] = r("tmin", 7, "min")
        f["wx_rain7"] = r("rain", 7, "sum")
        f["wx_rain14"] = r("rain", 14, "sum")
        f["wx_sun7"] = r("sun", 7, "sum")
        f["wx_frost7"] = (s["gmin"] <= 0).astype(float).where(s["gmin"].notna()).rolling(7, min_periods=4).sum()
        f["wx_wind7"] = r("wind", 7, "max")
        f["wx_hot7"] = (s["tmax"] >= 30).astype(float).where(s["tmax"].notna()).rolling(7, min_periods=4).sum()
        f["item"] = item
        out.append(f.rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "item", *WX_FEATS])


def attach(feat: pd.DataFrame, daily: pd.DataFrame | None) -> pd.DataFrame:
    """가격 피처에 날씨 피처를 붙인다. 날씨가 없으면 전부 NaN."""
    feat = feat.copy()
    if daily is None or daily.empty:
        for c in WX_FEATS:
            feat[c] = np.nan
        return feat
    wf = item_features(daily, sorted(feat["item"].unique()))
    return feat.merge(wf, on=["item", "date"], how="left")
