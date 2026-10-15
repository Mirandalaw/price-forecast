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
    assert set(res.model) == {"persist", "drift_m", "lgbm", "blend"}
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
