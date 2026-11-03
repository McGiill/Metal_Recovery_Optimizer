"""Grid Search (필수).

소유자: P4  (이 파일은 P4만 수정한다)

기본은 현장 단위 격자 (10 L × 1 ℃ × 0.5 hr ≈ 9.3만 점) 를 한 번에 평가한다.
격자 자체가 현장 단위라 반올림 · 재검증 단계가 필요 없다.
step_key="coarse_step" 은 비교 실험용 성긴 격자.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from module3_optimization.optimizers.base import OptResult, select_best

if TYPE_CHECKING:
    from module3_optimization.problem import Problem


def axis_values(spec: dict, step: float) -> np.ndarray:
    n = int(np.floor((spec["max"] - spec["min"]) / step + 1e-9))
    return np.round(spec["min"] + step * np.arange(n + 1), 6)


def make_grid(variables: dict, step_key: str = "unit") -> pd.DataFrame:
    names = list(variables)
    axes = [axis_values(variables[v], variables[v][step_key]) for v in names]
    mesh = np.meshgrid(*axes, indexing="ij")
    return pd.DataFrame({v: m.ravel() for v, m in zip(names, mesh)})


class GridSearch:
    name = "grid"

    def __init__(self, step_key: str = "unit"):
        self.step_key = step_key

    def run(
        self, problem: Problem, budget: int | None = None, seed: int = 0
    ) -> OptResult:
        """budget · seed 는 인터페이스 통일용이며 Grid 에서는 쓰지 않는다."""
        start = time.perf_counter()
        history = problem.evaluate(make_grid(problem.variables, self.step_key))
        history["eval_order"] = np.arange(len(history))
        best = select_best(history, problem.cfg["tie_break"])
        return OptResult(
            history=history,
            best=best,
            n_evals=len(history),
            elapsed_s=time.perf_counter() - start,
            algorithm=self.name
            if self.step_key == "unit"
            else f"{self.name}:{self.step_key}",
        )
