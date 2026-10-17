# price-forecast

2026 AI 농산물 가격예측 해커톤 — 배추·무·양파·깐마늘·건고추 10일 일별 + 순별 예측 파이프라인.

## 실행
```bash
pip install -r requirements.txt
# data/ 에 공공데이터포털 '주요 채소류 일일가격 현황' CSV(cp949)를 at5_raw_*.csv 로 넣는다
python scripts/run_backtest.py                 # 2021~2024 본선 동기간(10/26~11/09) 워크포워드, h=1..10 + 2순·3순 후
python scripts/predict.py --asof 2024-10-26    # 기준일 하나의 제출용 예측 → output/forecast_2024-10-26.csv
python scripts/make_submission.py --asof 2024-10-26   # 회차 제출: 예측 → 검증(60행·NaN·±40% 클리핑) → submission_*.csv + manifest_*.json
bash reproduce.sh 2024-10-26                  # 폐쇄망 재현: 고정 패키지(requirements-lock.txt) → 테스트 → 백테스트 → 제출 파일 + sha256
```

## 구조
- `price_forecast/loader.py`   CSV → (item, date, price, 순가격·전순·전년순·평년순). 거래일만 남김. 본선 데이터셋은 여기만 교체
- `price_forecast/features.py` 래그 변화율, MA 괴리, 변동성, 요일·연중일, 순별 비율, 계절 drift(leave-one-year-out), 타깃 y1..y10
- `price_forecast/models.py`   persistence / drift / LightGBM(예측일별, L1 + 1/가격 가중) / blend(수축 + 품목별 1·99% 클리핑)
- `price_forecast/sun.py`      순(旬) 정의·순 평균·2순/3순 후 타깃(ys2/ys3). 순 집계 방식이 바뀌면 `sun_mean()`만 교체
- `price_forecast/predict.py`  기준일 하나 → 5품목 × (1~10거래일, s2, s3) 예측. 모델 실패 시 품목·h 단위로 persistence 폴백
- `price_forecast/backtest.py` 연도별 워크포워드 + MAPE 리포트 + 품목·h별 모델 선택(`select`) + MAPE 수축 계수(`blend_k`) — 둘 다 이전 연도 성적만 사용
- `price_forecast/submission.py` 제출 파일 검증(완결성·범위 클리핑·flag)과 재현 매니페스트(입력 해시·버전·git)
- `data/at5_trading.csv`       정리된 거래일 가격 (2019-10 ~ 2024-12)
- `data/backtest_results.csv`  품목·연도·h·모델별 MAPE

## 결과 요약 (5품목 평균 MAPE, 각 연도 10/26~11/09 기준일)

두 가지 가격 자료로 검증한다. 둘 다 KAMIS 도매 조사가격이며, 본선 채점 대상(경락가)은 아니다.

| 자료 | 기간 | 검증 연도 | 설명 |
|---|---|---|---|
| `at5` (기본) | 2019-10~2024-12 | 2021~2024 (4년) | '주요 채소류 일일가격 현황' 전국 일일가격 — 기술서 수치의 근거 |
| `market` | 2015-01~2025-08 | 2017~2024 (8년) | '일자별 도소매 가격' 시장별 조사가격을 5품목 도매 상품 kg당 평균으로 집계 (`scripts/build_kamis_market.py`) |

`python scripts/run_backtest.py --source market --years 2017,2018,2019,2020,2021,2022,2023,2024`

**10거래일 앞**

| 자료 | persist | lgbm | lgbm_news | lgbm_wx | **auto** (배추·무 lgbm_news, 나머지 persist) |
|---|---|---|---|---|---|
| at5 4년 | 9.5 | 9.0 | 8.7 | 9.0 | **8.4** |
| market 8년 | 7.2 | 7.3 | 7.1 | 7.5 | **6.5** |

**2순 후 순평균**

| 자료 | persist | lgbm | lgbm_news | auto |
|---|---|---|---|---|
| at5 4년 | 10.5 | 11.0 | 10.9 | 9.9 |
| market 8년 | 8.2 | 8.1 | 8.1 | 7.5 |

- 배추·무가 오차의 대부분(10일 앞 12~20%). 양파·깐마늘·건고추는 1~4%로 기준가 유지(persist)가 가장 낫다.
- **auto = 배추·무는 뉴스 피처 LightGBM, 양파·깐마늘·건고추는 기준가 유지.** `make_submission.py` 기본값. 뉴스 신호 파일이 없으면 배추·무는 가격 피처만의 lgbm 으로 내려간다.
- 연도별로는 고르지 않다(market 8년, 배추 10일 앞 lgbm_news vs persist): 2017 25.0 vs 10.5, 2019 15.5 vs 21.0, 2021 27.8 vs 24.9, 2023 7.9 vs 19.6, 2024 6.0 vs 19.2.
  학습 연도가 2개뿐인 2017~2018 은 모델이 나쁘고, 학습 자료가 쌓인 2022~2024 는 꾸준히 앞선다. 2026 본선은 학습 연도 11개.
- 주의: 품목 배정은 같은 백테스트를 보고 정했으므로 실제 성능은 표보다 약간 나쁠 수 있다.

### 백테스트 방법 수정 (2026-10-18)
`seasonal_drift`(과거 연도 같은 시기 평균 변화율)를 전체 기간 leave-one-year-out 으로 만들던 것을 **학습 구간 연도만** 쓰도록 고쳤다.
이전 방식은 평가 연도보다 뒤 연도의 정보가 피처에 섞였다. 수정 후 at5 수치: lgbm_wx 8.8→9.0, auto(날씨) 8.4→8.5, lgbm_news 8.9→8.7.
**날씨 피처의 이득은 사라졌고 뉴스 피처의 이득이 남았다** → auto 의 배추·무 모델을 lgbm_wx 에서 lgbm_news 로 바꿈.

## 뉴스 피처 (news_collector 연동)
```bash
# news_collector 에서
python scripts/run_agri_signals.py --from-snapshots snapshots/ --out agri_signals.csv
cp agri_signals.csv ../price-forecast/data/
# price-forecast 에서 — 파일이 있으면 lgbm_news / blend_news 가 리포트에 추가됨 (뉴스 on/off 비교)
python scripts/run_backtest.py
```
`price_forecast/news.py`: 당일·3일·7일 롤링 방향 합, 사건군별 7일 카운트, news_shock(60일 90분위), 거시 7일,
그리고 이벤트 스터디 보정(`event_study` / `event_adjustment` → 모델 `blend_evt`). 과거 뉴스가 없으면 전부 0 → 가격 모델과 동일.

### 뉴스 on/off 실측 (2021~2024 소급 코퍼스 245건, `data/agri_signals.csv`)
market 자료 10일 앞, 뉴스가 있는 2022~2024 년 (lgbm → lgbm_news): 배추 14.3→10.8, 9.6→7.9, 10.3→6.0 / 무 14.1→11.2, 9.5→7.7, 22.6→23.1.
뉴스가 없는 2017~2021 은 두 모델이 같다(피처 전부 0). 이벤트 스터디 가산 보정(blend_evt)은 이득 없음 → 기본 off.
- 이벤트 스터디 자체는 뚜렷하다: 배추 **공급 증가 사건**(비축 방출·수입·할인 지원) 후 20일 −33% vs 가을 무조건 평균 −12.5%, 방향 적중 61~68%.
  반면 **공급 감소 사건**(폭염·태풍·김장 수요) 후에는 오히려 하락(이미 급등한 뒤 보도) → 역방향 신호. 즉 뉴스는 "대책이 나왔다 = 꼭지"를 알려주는 쪽이 유용하다.

### 시도했지만 기본 off 인 것
- `blend_k` MAPE 수축 계수: 연도마다 국면이 달라 계수가 뒤집힘. 본선 회차가 쌓여 같은 해 자료로 튜닝할 수 있을 때 다시 본다.
- `blend_evt` 이벤트 스터디 가산 보정: 위 뉴스 절 참조.
- 주산지 날씨 `lgbm_wx`/`blend_wx` (`price_forecast/weather.py`): 기상청 ASOS 시간자료(대관령·해남·의성, 2019-10~2024-12) → 일자료 → 품목별 7·14일 롤링. drift 누수 제거 후 lgbm 과 차이 없음(at5 9.0 vs 9.0, market 7.5 vs 7.3). 피처는 유지하되 auto 에서는 제외.

## 다음
1. 반입량 피처 (본선 데이터셋 R01·R02) — 배추·무 10일 오차의 핵심
2. 본선 데이터셋 로더(`load_auction`) + 제출 형식 맞춤 (10/26 OT 이후)
3. 2025-09 이후 시장별 자료가 포털에 올라오면 `market` 갱신 → 2025 fold
4. 순별(s2·s3) 개선: 재배면적·수입 피처
