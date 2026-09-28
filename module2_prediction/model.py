"""Module 2 — 예측 모델 인터페이스.

소유자: P3  (이 파일은 P3만 수정한다)
계약: docs/INTERFACE.md  ★ predict() 시그니처는 P4·P5가 의존한다. 변경 금지.
"""

import numpy as np
import pandas as pd


class RecoveryModel:
    """Ni·Co 회수율 동시 예측 모델."""

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "RecoveryModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """shape (n, 2) 배열을 반환한다. 열 순서는 [ni_recovery_pct, co_recovery_pct]."""
        raise NotImplementedError
