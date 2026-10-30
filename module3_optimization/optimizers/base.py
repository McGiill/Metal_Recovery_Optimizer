"""최적화기 공통 인터페이스 · 결과 구조 · 해 선택 규칙.

소유자: P4  (이 파일은 P4만 수정한다)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from module3_optimization.problem import Problem


@dataclass
class OptResult:
    history: pd.DataFrame  # 평가한 모든 점 + 예측 + 제약 + cost + eval_order
    best: pd.Series | None  # 실행 가능 해 중 최선. 없으면 None
    n_evals: int
    elapsed_s: float
    algorithm: str

    @property
    def found(self) -> bool:
        return self.best is not None


class Optimizer(Protocol):
    name: str

    def run(
        self, problem: Problem, budget: int | None = None, seed: int = 0
    ) -> OptResult: ...


def rank_feasible(history: pd.DataFrame, tie_break: list[str]) -> pd.DataFrame:
    """실행 가능 해를 동점 처리 순서대로 정렬 (모두 작은 쪽 우선). 없는 컬럼은 건너뛴다."""
    feasible = history[history["feasible"]]
    keys = [k for k in tie_break if k in feasible.columns]
    return feasible.sort_values(keys, kind="mergesort") if keys else feasible


def select_best(history: pd.DataFrame, tie_break: list[str]) -> pd.Series | None:
    ranked = rank_feasible(history, tie_break)
    return None if ranked.empty else ranked.iloc[0]


def select_closest(history: pd.DataFrame) -> pd.Series | None:
    """목표 달성 불가 시 대체 해: 목표 외 제약은 만족하면서 ① 미달량 합 최소 ② 비용 최소."""
    candidates = history[history["hard_ok"]]
    if candidates.empty:
        return None
    return candidates.sort_values(["shortfall_total", "cost"], kind="mergesort").iloc[0]
