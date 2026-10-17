import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from price_forecast import loader


def test_market_loader_schema_and_sun_columns():
    d = loader.load_prices("data", source="market")
    assert set(d["item"]) == set(loader.ITEMS)
    assert list(d.columns)[:7] == ["item", "date", "price", "p_sun", "p_prev_sun", "p_ly_sun", "p_normal_sun"]
    assert {"mk_qspread", "mk_garak", "mk_disp", "mk_nkinds", "mk_kindage"} <= set(d.columns)
    assert (d["price"] > 0).all() and d["p_sun"].notna().all()
    assert d["date"].min().year == 2015 and d["date"].max().year >= 2025
    # 평년은 직전 5개년 같은 순 평균의 최대·최소 제외 평균 → 자료 시작 2년 이후부터 값이 있어야 한다
    b = d[(d["item"] == "배추") & (d["date"] >= "2017-02-01")]
    assert b["p_normal_sun"].notna().all()
    # 순 평균은 그 순의 당일까지 평균: 순 첫날에는 당일 가격과 같다
    first = d[d["item"] == "무"].groupby(d["date"].dt.to_period("M").astype(str) + (d["date"].dt.day > 10).astype(str) + (d["date"].dt.day > 20).astype(str)).head(1)
    assert np.allclose(first["price"], first["p_sun"])
