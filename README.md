# price-forecast

2026 AI 농산물 가격예측 해커톤 — 배추·무·양파·깐마늘·건고추 10일 일별 + 순별 예측 파이프라인.

## 실행
```bash
pip install -r requirements.txt
# data/ 에 공공데이터포털 '주요 채소류 일일가격 현황' CSV(cp949)를 at5_raw_*.csv 로 넣는다
python scripts/run_backtest.py                 # 2021~2024 본선 동기간(10/26~11/09) 워크포워드, h=1..10 + 2순·3순 후
python scripts/predict.py --asof 2024-10-26    # 기준일 하나의 제출용 예측 → output/forecast_2024-10-26.csv
```

## 구조
- `price_forecast/loader.py`   CSV → (item, date, price, 순가격·전순·전년순·평년순). 거래일만 남김. 본선 데이터셋은 여기만 교체
- `price_forecast/features.py` 래그 변화율, MA 괴리, 변동성, 요일·연중일, 순별 비율, 계절 drift(leave-one-year-out), 타깃 y1..y10
- `price_forecast/models.py`   persistence / drift / LightGBM(예측일별, L1 + 1/가격 가중) / blend(수축 + 품목별 1·99% 클리핑)
- `price_forecast/sun.py`      순(旬) 정의·순 평균·2순/3순 후 타깃(ys2/ys3). 순 집계 방식이 바뀌면 `sun_mean()`만 교체
- `price_forecast/predict.py`  기준일 하나 → 5품목 × (1~10거래일, s2, s3) 예측. 모델 실패 시 품목·h 단위로 persistence 폴백
- `price_forecast/backtest.py` 연도별 워크포워드 + MAPE 리포트 + 품목·h별 모델 선택(`select`, 이전 연도 성적만 사용)
- `data/at5_trading.csv`       정리된 거래일 가격 (2019-10 ~ 2024-12)
- `data/backtest_results.csv`  품목·연도·h·모델별 MAPE

## 결과 (5품목 평균 MAPE, 2021~2024 각 10/26~11/09 기준일, KAMIS 조사가격)
| 지평 | persist | LightGBM | blend |
|---|---|---|---|
| 3거래일 | 6.1 | 5.5 | 5.4 |
| 10거래일 | 9.5 | 9.0 | 9.1 |
| 2순 후 순평균 | 10.5 | 11.4 | 11.2 |
| 3순 후 순평균 | 11.7 | 12.7 | 12.2 |

배추·무가 오차의 대부분(10일 앞 18~22%). 양파·깐마늘·건고추는 1~4%로 persistence가 충분. 10일 앞·순별은 가격 피처만으론 persistence를 못 이김 → 반입량·날씨·뉴스·면적 피처가 필요한 구간.

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
| 지평 | lgbm | lgbm_news | blend | blend_evt |
|---|---|---|---|---|
| 3거래일 | 5.5 | 5.6 | 5.4 | 5.5 |
| 10거래일 | 9.0 | **8.9** (무 19.9→19.2) | 9.1 | 9.4 |
| 2순 후 | 11.4 | 11.4 | 11.2 | 12.0 |

- 뉴스 피처(lgbm_news)는 10일 앞에서 소폭 개선, 저장품은 변화 없음. 이벤트 스터디 가산 보정(blend_evt)은 이득 없음 → 기본 off.
- 이벤트 스터디 자체는 뚜렷하다: 배추 **공급 증가 사건**(비축 방출·수입·할인 지원) 후 20일 −33% vs 가을 무조건 평균 −12.5%, 방향 적중 61~68%.
  반면 **공급 감소 사건**(폭염·태풍·김장 수요) 후에는 오히려 하락(이미 급등한 뒤 보도) → 역방향 신호. 즉 뉴스는 "대책이 나왔다 = 꼭지"를 알려주는 쪽이 유용하다.
- 10/26~11/09 평가 구간은 계절 하락이 지배적이라 가격 피처만으로도 상당 부분 설명됨. 뉴스 효과는 9~10월 급등 국면에서 더 클 것으로 본다(미검증).

## 다음
1. 반입량 피처 (본선 데이터셋 R01·R02) — 배추·무 10일 오차의 핵심
2. MAPE 맞춤 수축 계수 (품목별, 과거 연도 튜닝)
3. 주산지 날씨 피처 (ASOS)
4. 2021~2024 가을 뉴스 소급 → `agri_signals.csv` → 뉴스 on/off 실측
5. 2025 파일 추가 → 2025 fold
6. 본선 데이터셋 로더(`load_auction`) + 제출 형식 맞춤 + `reproduce.sh`
