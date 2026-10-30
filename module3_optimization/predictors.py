"""예측기 인터페이스 · 가짜 모델 · 어댑터.

소유자: P4  (이 파일은 P4만 수정한다)

Module 3 내부의 모든 예측기는 Predictor 형태로 맞춘다.
  입력: 여러 행 DataFrame (배치)
  출력: 같은 행 수 · 같은 index, 컬럼 ni_pred, co_pred, ni_std, co_std

P3 의 RecoveryModel.predict() 는 ndarray (n, 2) 를 반환하므로
RecoveryModelAdapter 로 감싸서 쓴다 (P3 시그니처는 변경 금지).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
import pandas as pd

PRED_COLS = ["ni_pred", "co_pred", "ni_std", "co_std"]
TARGET_COLS = ["ni_recovery_pct", "co_recovery_pct"]


@runtime_checkable
class Predictor(Protocol):
    name: str

    def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...


def _frame(index: pd.Index, ni, co, ni_std, co_std) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ni_pred": np.asarray(ni, dtype=float),
            "co_pred": np.asarray(co, dtype=float),
            "ni_std": np.asarray(ni_std, dtype=float),
            "co_std": np.asarray(co_std, dtype=float),
        },
        index=index,
    )


class MockAnalyticPredictor:
    """개발·검증용 해석적 가짜 모델. 결과 수치로 쓰지 않는다.

    가산형 포화 모델 (상수는 analysis/calibrate_mock.py 로 원본 데이터에 맞춤):

        회수율 = b0 + b_acid·g(산 비율, s_acid) + b_temp·g(온도 − 40, s_temp)
                    + b_time·g(시간, s_time) − b_imp·불순물 합,   g(x, s) = 1 − exp(−x/s)

    - 결정 변수마다 증가하다가 둔해진다 (원본 잔차가 가운데 +, 양 끝 − 인 패턴)
    - 100% 로 자른다
    - 결정 변수에 대해 단조 비감소 → 최소 황산량 정답을 이분 탐색으로 구할 수 있다
    - σ 는 상수 (보정 잔차 RMSE)
    """

    name = "mock_analytic"

    DEFAULT_NI: ClassVar[dict] = {
        "b0": 60.94,
        "acid": 19.93,
        "temp": 11.57,
        "time": 14.36,
        "impurity": 0.8919,
    }
    DEFAULT_CO: ClassVar[dict] = {
        "b0": 55.59,
        "acid": 19.52,
        "temp": 13.65,
        "time": 13.35,
        "impurity": 1.254,
    }

    def __init__(
        self,
        acid_scale: float = 6.0,
        temp_scale: float = 29.16,
        time_scale: float = 3.154,
        temp_origin: float = 40.0,
        cap: float = 100.0,
        ni: dict | None = None,
        co: dict | None = None,
        std: dict | None = None,
    ):
        self.acid_scale = acid_scale
        self.temp_scale = temp_scale
        self.time_scale = time_scale
        self.temp_origin = temp_origin
        self.cap = cap
        self.ni = {**self.DEFAULT_NI, **(ni or {})}
        self.co = {**self.DEFAULT_CO, **(co or {})}
        self.std = {"ni": 1.62, "co": 1.84, **(std or {})}

    @staticmethod
    def _sat(x, scale):
        return 1.0 - np.exp(-np.asarray(x, dtype=float) / scale)

    def _recovery(self, X: pd.DataFrame, coef: dict) -> np.ndarray:
        impurity = X["cu_pct"] + X["fe_pct"] + X["al_pct"]
        value = (
            coef["b0"]
            + coef["acid"] * self._sat(X["acid_l"] / X["feed_kg"], self.acid_scale)
            + coef["temp"] * self._sat(X["temp_c"] - self.temp_origin, self.temp_scale)
            + coef["time"] * self._sat(X["time_hr"], self.time_scale)
            - coef["impurity"] * impurity
        )
        return np.minimum(np.asarray(value, dtype=float), self.cap)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        return _frame(
            X.index,
            self._recovery(X, self.ni),
            self._recovery(X, self.co),
            np.full(n, self.std["ni"]),
            np.full(n, self.std["co"]),
        )


class RecoveryModelAdapter:
    """P3 RecoveryModel (predict → ndarray (n, 2), 열 순서 [ni, co]) 을 Predictor 로 감싼다.

    불확실성은 std_fn → model.predict_std → 0 순서로 찾는다.
    """

    def __init__(
        self,
        model,
        std_fn: Callable[[pd.DataFrame], np.ndarray] | None = None,
        name: str | None = None,
    ):
        self.model = model
        self.std_fn = std_fn
        self.name = name or f"real:{type(model).__name__}"

    @property
    def has_std(self) -> bool:
        return self.std_fn is not None or hasattr(self.model, "predict_std")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        mean = np.asarray(self.model.predict(X), dtype=float)
        if mean.shape != (len(X), 2):
            raise ValueError(
                f"predict() 는 shape ({len(X)}, 2) 를 반환해야 한다. 받은 shape: {mean.shape}"
            )

        if self.std_fn is not None:
            std = self.std_fn(X)
        elif hasattr(self.model, "predict_std"):
            std = self.model.predict_std(X)
        else:
            std = np.zeros_like(mean)
        std = np.asarray(std, dtype=float)
        if std.shape != mean.shape:
            raise ValueError(
                f"std shape {std.shape} 가 예측 shape {mean.shape} 와 다르다."
            )

        return _frame(X.index, mean[:, 0], mean[:, 1], std[:, 0], std[:, 1])


class ProxyRFPredictor:
    """개발용 임시 대리 모델 (결측 없는 행만으로 학습한 RandomForest 2개).

    최종 산출물이 아니다. artifacts/ 에만 저장하고 커밋하지 않는다.
    std 는 트리별 예측의 표준편차.
    """

    name = "proxy_rf"

    DEFAULT_FEATURES = (
        "ni_pct", "co_pct", "li_pct", "mn_pct",
        "cu_pct", "fe_pct", "al_pct",
        "feed_kg", "acid_l", "temp_c", "time_hr",
    )  # fmt: skip

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        n_estimators: int = 300,
        random_state: int = 0,
    ):
        self.feature_cols = list(feature_cols or self.DEFAULT_FEATURES)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models_: dict = {}

    def fit(self, df: pd.DataFrame) -> ProxyRFPredictor:
        from sklearn.ensemble import RandomForestRegressor

        data = df.dropna(subset=self.feature_cols + TARGET_COLS)
        X = data[self.feature_cols].to_numpy(dtype=float)
        for target in TARGET_COLS:
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators, random_state=self.random_state
            )
            self.models_[target] = rf.fit(X, data[target].to_numpy(dtype=float))
        self.n_train_ = len(data)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("fit() 을 먼저 호출해야 한다.")
        values = X[self.feature_cols].to_numpy(dtype=float)
        stats = {}
        for target, rf in self.models_.items():
            per_tree = np.stack([tree.predict(values) for tree in rf.estimators_])
            stats[target] = (per_tree.mean(axis=0), per_tree.std(axis=0))
        (ni, ni_std), (co, co_std) = stats["ni_recovery_pct"], stats["co_recovery_pct"]
        return _frame(X.index, ni, co, ni_std, co_std)


def build_predictor(cfg: dict) -> Predictor:
    """config 의 predictor.type 에 따라 예측기를 만든다."""
    pcfg = cfg["predictor"]
    kind = pcfg["type"]

    if kind == "mock_analytic":
        return MockAnalyticPredictor(**(pcfg.get("mock") or {}))

    if kind in ("proxy_rf", "real"):
        path = pcfg.get("path")
        if not path:
            raise ValueError(f"predictor.type={kind} 에는 predictor.path 가 필요하다.")
        import joblib

        obj = joblib.load(path)
        if kind == "proxy_rf":
            return obj
        return RecoveryModelAdapter(obj, name=pcfg.get("version") or None)

    raise ValueError(f"알 수 없는 predictor.type: {kind}")
