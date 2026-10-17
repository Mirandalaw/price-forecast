"""합성 데이터로 로더 → 피처 → 모델 → 백테스트가 끝까지 도는지 확인 (외부 파일·네트워크 불필요)."""
import numpy as np
import pandas as pd
import pytest

from price_forecast import backtest, features as F, loader, models as M


def _synthetic(seed=0, years=(2019, 2020, 2021, 2022, 2023, 2024)):
    rng = np.random.default_rng(seed)
    rows = []
    for item, base in [("배추", 1000), ("무", 900), ("양파", 800), ("깐마늘", 7000), ("건고추", 15000)]:
        for yr in years:
            days = pd.bdate_range(f"{yr}-01-01", f"{yr}-12-31")
            lp = np.log(base) + np.cumsum(rng.normal(0, 0.02, len(days)))
            # 10월 하순~11월 초 하락 계절성
            season = np.where((days.month == 10) & (days.day >= 20) | (days.month == 11) & (days.day <= 10), -0.004, 0)
            lp = lp + np.cumsum(season)
            p = np.exp(lp)
            for d, v in zip(days, p):
                rows.append(dict(item=item, date=d, price=v, p_sun=v * 1.01, p_prev_sun=v * 1.02,
                                 p_ly_sun=v * 0.95, p_normal_sun=v))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def prices():
    return _synthetic()


def test_features_have_targets_and_no_future_leak(prices):
    feat = F.build(prices)
    assert {"y1", "y10", "p10", "lag_ret1", "ma7_gap", "sun_gap"} <= set(feat.columns)
    g = feat[feat.item == "배추"].sort_values("date")
    # y1 = 다음 행 logp − 현재 logp
    i = 50
    assert np.isclose(g.y1.iloc[i], g.logp.iloc[i + 1] - g.logp.iloc[i])
    # 마지막 10행은 타깃이 없어야 함
    assert g.y10.tail(10).isna().all()


def test_seasonal_drift_excludes_own_year(prices):
    feat = F.build(prices)
    d = F.seasonal_drift(feat, h=10)
    assert d.notna().mean() > 0.5
    # 같은 날짜·품목에서 연도만 다르면 drift 값이 달라야 함 (자기 연도 제외)
    g = feat[(feat.item == "무") & (feat.doy == 300)]
    assert g.index.size >= 3 and d[g.index].nunique() >= 2


def test_lgbm_weights_and_clip(prices):
    feat = F.build(prices); feat["drift"] = F.seasonal_drift(feat, 5)
    train = feat[feat.date < "2024-10-26"]; test = feat[(feat.date >= "2024-10-26") & (feat.date <= "2024-11-09")].dropna(subset=["p5"])
    pr = M.predict_all(train, test, h=5)
    assert {"persist", "drift_m", "lgbm", "blend"} <= set(pr.columns)
    assert (pr.persist == 0).all()
    lo = train.groupby("item").y5.quantile(0.01); hi = train.groupby("item").y5.quantile(0.99)
    for item, s in pr.groupby("item").blend:
        assert (s >= lo[item] - 1e-9).all() and (s <= hi[item] + 1e-9).all()


def test_backtest_runs_and_beats_nothing_absurd(prices):
    res = backtest.run(prices, years=(2023, 2024), horizons=(1, 5))
    assert set(res.model) == {"persist", "drift_m", "lgbm", "blend", "blend_k"}
    assert (res.mape >= 0).all() and (res.mape < 100).all()
    rep = backtest.report(res)
    assert "5품목 평균" in rep


def test_loader_garlic_filter(tmp_path):
    # 2023형(품종코드 있음): 마늘 두 품종 중 12A9만 남아야 함
    cols = ["조회일자", "품목코드", "품목명", "품종코드", "도매일일가격", "도매순가격", "도매전순가격", "도매전년순가격", "도매평년순가격", "도매일자"]
    rows = [["2023-01-02", "1209", "마늘", "12A9", "7000", "7000", "6900", "6500", "6800", "2023-01-02"],
            ["2023-01-02", "1209", "마늘", "1209", "5000", "5000", "4900", "4500", "4800", "2023-01-02"],
            ["2023-01-03", "1209", "마늘", "12A9", "7100", "7000", "6900", "6500", "6800", "2023-01-03"],
            ["2023-01-02", "1001", "배추", "1001", "1000", "1000", "990", "950", "980", "2023-01-02"]]
    p = tmp_path / "at5_raw_test.csv"
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False, encoding="cp949")
    df = loader.load_at5([p])
    assert set(df.item) == {"깐마늘", "배추"}
    assert (df[df.item == "깐마늘"].price > 6000).all()


def test_news_features_attach_and_backtest_with_news(prices):
    from price_forecast import news as N
    # 합성 신호: 배추에 2024-10-28 비축 방출(−), 2024-11-02 폭우(+)
    sig = pd.DataFrame([
        dict(date=pd.Timestamp("2024-10-28"), item="배추", news_cnt=3, event_cnt=1, dir_sum=-1, dir_wsum=-1.25,
             evt_agri_reserve_release=1, mag_ton=10000, macro_fx_up=0, macro_oil_up=0),
        dict(date=pd.Timestamp("2024-11-02"), item="배추", news_cnt=2, event_cnt=1, dir_sum=1, dir_wsum=0.5,
             evt_agri_heavy_rain=1, mag_ton=0, macro_fx_up=1, macro_oil_up=0),
    ]).fillna(0)
    feat = N.attach(F.build(prices), sig)
    assert set(N.NEWS_FEATS) <= set(feat.columns)
    row = feat[(feat.item == "배추") & (feat.date == "2024-10-28")].iloc[0]
    assert row.nw_dir == -1 and row.nw_release == 1
    later = feat[(feat.item == "배추") & (feat.date == "2024-11-01")].iloc[0]
    assert later.nw_release == 1 and later.nw_wdir_7d == -1.25      # 7일 롤링에 남아 있음
    assert feat[feat.item == "양파"][N.NEWS_FEATS].drop(columns=["mc_fx_7d", "mc_oil_7d"]).abs().sum().sum() == 0
    res = backtest.run(prices, years=(2024,), horizons=(3,), signals=sig)
    assert {"lgbm_news", "blend_news"} <= set(res.model)


# ---------------------------------------------------------------------------
# 순별 타깃 · 제출용 예측
# ---------------------------------------------------------------------------
from price_forecast import predict, sun  # noqa: E402


def test_sun_targets_match_sun_mean(prices):
    feat = F.build(prices)
    g = feat[feat.item == "무"].set_index("date")
    t = pd.Timestamp("2022-10-05")                      # 10월 상순 → s2 = 10월 하순, s3 = 11월 상순
    p = prices[prices.item == "무"].set_index("date")["price"]
    exp_s2 = p["2022-10-21":"2022-10-31"].mean()
    exp_s3 = p["2022-11-01":"2022-11-10"].mean()
    assert np.isclose(g.loc[t, "ps2"], exp_s2) and np.isclose(g.loc[t, "ps3"], exp_s3)
    assert np.isclose(g.loc[t, "ys2"], np.log(exp_s2) - np.log(g.loc[t, "price"]))
    assert sun.target_label(t, "s2") == "2022-10 하순" and sun.target_label(t, "s3") == "2022-11 상순"


def test_backtest_includes_sun_horizons(prices):
    res = backtest.run(prices, years=(2023,), horizons=[1, "s2"])
    assert set(res["h"].astype(str)) == {"1", "s2"}
    assert "select" in set(backtest.add_selection(res)["model"])


def test_forecast_complete_and_uses_only_past(prices):
    fc = predict.forecast(prices, "2023-10-26", horizons=[1, 10, "s2", "s3"])
    assert len(fc) == 5 * 4 and fc["pred_price"].notna().all() and (fc["pred_price"] > 0).all()
    assert set(fc["target"][fc.horizon == "s2"]) == {"2023-11 중순"}
    # 기준가는 2023-10-26 이하 마지막 거래일 가격이어야 한다
    last = prices[(prices.item == "배추") & (prices.date <= "2023-10-26")].sort_values("date").iloc[-1]
    assert np.isclose(fc[(fc.item == "배추")]["base_price"].iloc[0], round(last.price, 1))


def test_forecast_falls_back_to_persistence(prices, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("lightgbm down")
    monkeypatch.setattr(M, "predict_all", boom)
    fc = predict.forecast(prices, "2023-10-26", horizons=[3, "s2"])
    assert (fc["model"] == "persist(fallback)").all()
    assert np.allclose(fc["pred_price"], fc["base_price"])


# ---------------------------------------------------------------------------
# 재현성 · 누출 · 제출 검증
# ---------------------------------------------------------------------------
from price_forecast import submission  # noqa: E402


def test_forecast_is_deterministic(prices):
    a = predict.forecast(prices, "2023-10-26", horizons=[1, 10, "s2"])
    b = predict.forecast(prices, "2023-10-26", horizons=[1, 10, "s2"])
    pd.testing.assert_frame_equal(a, b)


def test_forecast_ignores_prices_after_asof(prices):
    """기준일 이후 가격을 아무렇게나 바꿔도 예측이 같아야 한다 (미래 정보 누출 없음)."""
    asof = "2023-10-26"
    tampered = prices.copy()
    after = tampered["date"] > asof
    tampered.loc[after, "price"] = tampered.loc[after, "price"] * 3.0
    a = predict.forecast(prices, asof, horizons=[5, "s3"])
    b = predict.forecast(tampered, asof, horizons=[5, "s3"])
    pd.testing.assert_frame_equal(a, b)


def test_submission_validate_clips_and_flags(prices):
    fc = predict.forecast(prices, "2023-10-26")
    assert len(fc) == submission.EXPECTED_ROWS
    ok = submission.validate(fc)
    assert (ok["flag"] == "").all()
    # 한 행을 터무니없는 값으로 만들면 경계로 잘리고 flag 가 남는다
    bad = fc.copy()
    bad.loc[0, "pred_price"] = bad.loc[0, "base_price"] * 5
    v = submission.validate(bad)
    assert v.loc[0, "flag"].startswith("clipped_high") and np.isclose(v.loc[0, "pred_price"], bad.loc[0, "base_price"] * submission.HI)


def test_submission_rejects_incomplete(prices):
    fc = predict.forecast(prices, "2023-10-26")
    with pytest.raises(submission.SubmissionError):
        submission.validate(fc.iloc[:-1])
    nan = fc.copy()
    nan.loc[3, "pred_price"] = np.nan
    with pytest.raises(submission.SubmissionError):
        submission.validate(nan)


def test_submission_write_creates_files(prices, tmp_path):
    fc = predict.forecast(prices, "2023-10-26")
    out, man = submission.write(fc, "2023-10-26", "blend", [], tmp_path)
    assert out.exists() and man.exists()
    got = pd.read_csv(out)
    assert list(got.columns) == ["asof", "item", "horizon", "target", "pred_price"] and len(got) == submission.EXPECTED_ROWS


def test_mape_shrink_factor_prefers_under_prediction():
    # 실제가가 예측보다 자주 낮은 분포에서는 k < 1 이 선택된다
    rng = np.random.default_rng(0)
    pred = np.full(200, 100.0)
    actual = 100.0 * np.exp(rng.normal(-0.08, 0.15, 200))
    assert backtest.mape_shrink_factor(list(zip(pred, actual))) < 1.0
    assert backtest.mape_shrink_factor([]) == 1.0
