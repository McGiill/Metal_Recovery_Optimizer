"""최적화 알고리즘 모음. 소유자: P4"""

from module3_optimization.optimizers.base import (
    Optimizer,
    OptResult,
    select_best,
    select_closest,
)
from module3_optimization.optimizers.grid import GridSearch, make_grid

_REGISTRY = {
    "grid": lambda: GridSearch("unit"),
    "grid_coarse": lambda: GridSearch("coarse_step"),
    # 5단계: "random", "bayes", "ga"
}


def get_optimizer(name: str) -> Optimizer:
    if name not in _REGISTRY:
        raise ValueError(
            f"알 수 없는 알고리즘: {name} (사용 가능: {sorted(_REGISTRY)})"
        )
    return _REGISTRY[name]()


__all__ = [
    "GridSearch",
    "OptResult",
    "Optimizer",
    "get_optimizer",
    "make_grid",
    "select_best",
    "select_closest",
]
