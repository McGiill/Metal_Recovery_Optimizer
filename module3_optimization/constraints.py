"""제약 조건 함수.

소유자: P4  (이 파일은 P4만 수정한다)

모든 제약은 (만족 여부: bool Series, 위반량: float Series) 를 반환한다.
  - Grid · Random      → 만족 여부로 필터
  - GA · Optuna        → 위반량을 페널티로
  - NSGA-II            → 위반량을 G 로 (constraint domination)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ConstraintResult = tuple[pd.Series, pd.Series]

# 부동소수점 격자값(예: 400 + 10·i)이 경계에서 떨어지지 않도록 허용 오차를 둔다.
EPS = 1e-9


def bounded(
    values: pd.Series, lo: float | None = None, hi: float | None = None
) -> ConstraintResult:
    """lo ≤ values ≤ hi (경계 포함). None 인 쪽은 검사하지 않는다."""
    values = values.astype(float)
    violation = pd.Series(0.0, index=values.index)
    if lo is not None:
        violation += (lo - values).clip(lower=0)
    if hi is not None:
        violation += (values - hi).clip(lower=0)
    violation = violation.where(violation > EPS, 0.0)
    return violation == 0.0, violation


def box(df: pd.DataFrame, var: str, lo: float, hi: float) -> ConstraintResult:
    """결정 변수 상·하한."""
    return bounded(df[var], lo, hi)


def ratio(
    df: pd.DataFrame, col: str, lo: float | None, hi: float | None
) -> ConstraintResult:
    """비율 제약 (예: acid_per_feed). 범위가 둘 다 None 이면 항상 만족."""
    return bounded(df[col], lo, hi)


def target(df: pd.DataFrame, judged_col: str, minimum: float) -> ConstraintResult:
    """목표 회수율. judged_col 은 μ (평균 기준) 또는 μ − zσ (확률 제약 기준)."""
    return bounded(df[judged_col], lo=minimum)


def round_to_unit(U: pd.DataFrame, variables: dict) -> pd.DataFrame:
    """연속 알고리즘 결과를 현장 단위로 반올림하고 박스 안으로 자른다.

    격자 원점은 각 변수의 min 이다 (min + k·unit).
    """
    out = U.copy()
    for var, spec in variables.items():
        if var not in out:
            continue
        lo, hi, unit = spec["min"], spec["max"], spec["unit"]
        snapped = lo + np.round((out[var].astype(float) - lo) / unit) * unit
        max_on_grid = lo + np.floor((hi - lo) / unit + EPS) * unit
        out[var] = np.round(np.clip(snapped, lo, max_on_grid), 6)
    return out
