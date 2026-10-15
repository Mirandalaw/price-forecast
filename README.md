# price-forecast

2026 AI 농산물 가격예측 해커톤 — 배추·무·양파·깐마늘·건고추 10일 일별 + 순별 예측 파이프라인.

## 실행
```bash
pip install -r requirements.txt
# data/ 에 공공데이터포털 '주요 채소류 일일가격 현황' CSV(cp949)를 at5_raw_*.csv 로 넣는다
python scripts/run_backtest.py        # 2021~2024 본선 동기간(10/26~11/09) 워크포워드, h=1..10
```

## 구조
- `price_forecast/loader.py`   CSV → (item, date, price, 순가격·전순·전년순·평년순). 거래일만 남김. 본선 데이터셋은 여기만 교체
- `price_forecast/features.py` 래그 변화율, MA 괴리, 변동성, 요일·연중일, 순별 비율, 계절 drift(leave-one-year-out), 타깃 y1..y10
- `price_forecast/models.py`   persistence / drift / LightGBM(예측일별, L1 + 1/가격 가중) / blend(수축 + 품목별 1·99% 클리핑)
- `price_forecast/backtest.py` 연도별 워크포워드 + MAPE 리포트
- `data/at5_trading.csv`       정리된 거래일 가격 (2019-10 ~ 2024-12)
- `data/backtest_results.csv`  품목·연도·h·모델별 MAPE

## 첫 결과 (10일 앞, 5품목 평균 MAPE)
persistence 9.5 → LightGBM 9.0. 배추·무는 19~20%로 가격 피처만으론 거의 안 줄어듦 → 반입량·뉴스 사건·재배면적 피처가 필요한 구간.

## 다음
1. 2025년 파일 추가 → 2025 fold
2. 반입량 피처 (본선 데이터셋 또는 가락시장)
3. 뉴스 사건 피처 (news_collector agri 패치 + KREI 월보 소급 라벨)
4. 순별(2순·3순 후) 모델: p_sun 기준 유사사례 + 면적·수입
5. 회차 제출 파일 생성 + fallback
