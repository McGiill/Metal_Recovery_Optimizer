"""적용 범위 (외삽 방지) 판정.

소유자: P4  (이 파일은 P4만 수정한다)

  - 표준화 공간에서 k-최근접 학습 배치까지의 평균 거리
  - 임계값 = 학습 배치끼리의 같은 거리(자기 자신 제외)의 percentile 백분위
  - 두 개를 쓴다
      process: 결정 변수 + 원료 → 후보 공정 조건마다 판정 (Problem 의 domain 제약)
      feed   : 원료 성분만     → 원료 자체가 범위 밖이면 recommend() 가 out_of_domain

학습 데이터: 지금은 원본 최소 정리본(analysis/load_raw.py), P1 정제 데이터가 오면
config.yaml 의 domain.source 를 processed 로 바꾼다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from module3_optimization.config import get

DECISION_COLS = ["acid_l", "temp_c", "time_hr"]
FEED_COLS = [
    "ni_pct",
    "co_pct",
    "li_pct",
    "mn_pct",
    "cu_pct",
    "fe_pct",
    "al_pct",
    "feed_kg",
]


class DomainChecker:
    def __init__(self, name: str = "domain"):
        self.name = name

    def fit(
        self, X_train: pd.DataFrame, cols: list[str], k: int = 5, percentile: float = 95
    ) -> DomainChecker:
        data = X_train[cols].dropna()
        if len(data) <= k:
            raise ValueError(f"학습 행 수 {len(data)} 가 k={k} 보다 많아야 한다.")
        self.cols = list(cols)
        self.k = k
        self.percentile = percentile
        self.n_train = len(data)
        self.scaler = StandardScaler().fit(data.to_numpy(dtype=float))
        self._train_z = self.scaler.transform(data.to_numpy(dtype=float))
        self.nn = NearestNeighbors(n_neighbors=k + 1).fit(self._train_z)

        dist, _ = self.nn.kneighbors(self._train_z)
        self.train_distance_ = dist[:, 1:].mean(axis=1)  # 자기 자신 제외
        self.threshold = float(np.percentile(self.train_distance_, percentile))
        return self

    def _z(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.cols].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError(
                f"{self.name}: 거리 계산 컬럼에 NaN 이 있다 ({self.cols})."
            )
        return self.scaler.transform(values)

    def distance(self, X: pd.DataFrame) -> np.ndarray:
        """각 행의 k-최근접 학습 배치 평균 거리."""
        dist, _ = self.nn.kneighbors(self._z(X), n_neighbors=self.k)
        return dist.mean(axis=1)

    def in_domain(self, X: pd.DataFrame) -> np.ndarray:
        return self.distance(X) <= self.threshold

    def similar_count(self, X: pd.DataFrame, radius: float | None = None) -> np.ndarray:
        """반경 안 학습 배치 수 — 레시피 카드의 '유사 과거 배치 수'. 기본 반경 = threshold."""
        neighbors = self.nn.radius_neighbors(
            self._z(X), radius=radius or self.threshold, return_distance=False
        )
        return np.array([len(n) for n in neighbors])


def load_training_data(cfg: dict) -> pd.DataFrame:
    source = get(cfg, "domain.source", "raw_minimal")
    if source == "raw_minimal":
        from module3_optimization.analysis.load_raw import load_minimal

        return load_minimal()
    if source == "processed":
        path = get(cfg, "domain.path")
        if not path:
            raise ValueError("domain.source=processed 에는 domain.path 가 필요하다.")
        return pd.read_csv(path)
    raise ValueError(f"알 수 없는 domain.source: {source}")


def build_domains(
    cfg: dict, train: pd.DataFrame | None = None
) -> tuple[DomainChecker, DomainChecker]:
    """(process, feed) 판정기. train 을 안 주면 domain.source 에서 읽는다."""
    train = load_training_data(cfg) if train is None else train
    k = get(cfg, "domain.k", 5)
    process_cols = get(cfg, "domain.columns") or DECISION_COLS + FEED_COLS
    process = DomainChecker("process").fit(
        train, process_cols, k, get(cfg, "domain.percentile", 95)
    )
    feed = DomainChecker("feed").fit(
        train, FEED_COLS, k, get(cfg, "domain.feed_percentile", 99)
    )
    return process, feed
