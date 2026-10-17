#!/usr/bin/env bash
# 폐쇄망 재현용: 외부 호출 없이 (1) 테스트 (2) 백테스트 (3) 기준일 제출 파일을 다시 만든다.
#   bash reproduce.sh 2024-10-26
# 전제: data/at5_raw_*.csv (가격), data/agri_signals.csv (뉴스 신호, 선택) 가 함께 제출됨. 패키지는 requirements-lock.txt 로 고정.
set -euo pipefail
ASOF="${1:?usage: reproduce.sh YYYY-MM-DD}"
cd "$(dirname "$0")"
python3 -m venv .venv-repro && . .venv-repro/bin/activate
pip install -q --no-input -r requirements-lock.txt
python -m pytest -q
python scripts/run_backtest.py > output/backtest_report_${ASOF}.txt
python scripts/make_submission.py --asof "$ASOF"
echo "--- sha256 ---"; sha256sum output/submission_${ASOF}.csv output/manifest_${ASOF}.json
