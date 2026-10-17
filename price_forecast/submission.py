"""회차 제출 파일 생성 + 검증 + 재현 매니페스트.

제출 파일은 predict.forecast() 결과를 그대로 쓰되, 보내기 전에 세 가지를 확인한다.
  1. 완결성: 5품목 × 12지평 = 60행, 값은 전부 유한한 양수
  2. 범위:   기준가 대비 [LO, HI] 배 밖이면 경계로 자르고 flag 열에 기록 (한 회차를 통째로 망치지 않기 위한 마지막 안전장치)
  3. 재현성: 입력 파일 해시·모델·라이브러리 버전·git 커밋을 manifest JSON 으로 남긴다
본선 제출 형식이 정해지면 to_contest_format() 만 바꾼다.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F, loader

LO, HI = 0.6, 1.4          # 기준가 대비 허용 배수 (10거래일·순별 과거 1·99% 분위가 대략 ±35% 안)
EXPECTED_ROWS = len(loader.ITEMS) * len(F.ALL_HORIZONS)


class SubmissionError(ValueError):
    pass


def validate(fc: pd.DataFrame) -> pd.DataFrame:
    """검증 후 flag 열을 붙여 돌려준다. 복구 불가능한 문제(행 누락·NaN)는 예외."""
    fc = fc.copy()
    if len(fc) != EXPECTED_ROWS:
        raise SubmissionError(f"행 수 {len(fc)} ≠ {EXPECTED_ROWS}")
    if set(fc["item"]) != set(loader.ITEMS):
        raise SubmissionError(f"품목 누락: {set(loader.ITEMS) - set(fc['item'])}")
    if fc.groupby("item").size().nunique() != 1:
        raise SubmissionError("품목별 지평 수가 다름")
    bad = ~np.isfinite(fc["pred_price"]) | (fc["pred_price"] <= 0) | ~np.isfinite(fc["base_price"])
    if bad.any():
        raise SubmissionError(f"유효하지 않은 값 {int(bad.sum())}행")
    ratio = fc["pred_price"] / fc["base_price"]
    fc["flag"] = ""
    fc.loc[ratio < LO, "flag"] = f"clipped_low(<{LO}x)"
    fc.loc[ratio > HI, "flag"] = f"clipped_high(>{HI}x)"
    fc["pred_price"] = (fc["base_price"] * ratio.clip(LO, HI)).round(1)
    return fc


def to_contest_format(fc: pd.DataFrame) -> pd.DataFrame:
    """본선 제출 양식. 아직 미정이라 내부 형식 그대로 — 명세가 오면 여기만 바꾼다."""
    return fc[["asof", "item", "horizon", "target", "pred_price"]]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def manifest(asof: str, model: str, inputs: list[Path], out: Path, fc: pd.DataFrame) -> dict:
    import lightgbm, numpy, pandas, sklearn  # noqa: PLC0415

    return dict(
        asof=asof, model=model, git=_git_rev(), python=platform.python_version(),
        libs=dict(pandas=pandas.__version__, numpy=numpy.__version__, lightgbm=lightgbm.__version__, sklearn=sklearn.__version__),
        inputs={str(p): _sha(p) for p in inputs if Path(p).exists()},
        output={str(out): _sha(out)},
        n_rows=int(len(fc)), n_flagged=int((fc["flag"] != "").sum()), fallback_rows=int(fc["model"].str.contains("fallback").sum()),
    )


def write(fc: pd.DataFrame, asof: str, model: str, inputs: list[Path], out_dir: Path = Path("output")) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fc = validate(fc)
    out = out_dir / f"submission_{asof}.csv"
    to_contest_format(fc).to_csv(out, index=False)
    (out_dir / f"forecast_{asof}_full.csv").write_text(fc.to_csv(index=False), encoding="utf-8")
    m = manifest(asof, model, inputs, out, fc)
    mpath = out_dir / f"manifest_{asof}.json"
    mpath.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, mpath
