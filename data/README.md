# data/

- `at5_trading.csv`  거래일 기준 5품목 도매가격 (2019-10 ~ 2024-12). 출처: 공공데이터포털 '한국농수산식품유통공사_주요 채소류 일일가격 현황' (KAMIS 도매 조사가격). 마늘은 품종 12A9(깐마늘)만.
- `at5_daily.csv`    위와 같되 휴장일 포함 달력 기준 (조회일자 행 전부).
- `backtest_results.csv`  품목·연도·h·모델별 MAPE (scripts/run_backtest.py 출력).
- `at5_raw_*.csv`    원본 CSV. git에는 올리지 않음 — 포털에서 받아 이 이름으로 둔다.

채점 타깃인 경락가·반입량은 본선 표준 데이터셋(10/26)으로 받는다. 받으면 `data/raw/` 아래 두고 loader.py 에 load_auction() 추가.
