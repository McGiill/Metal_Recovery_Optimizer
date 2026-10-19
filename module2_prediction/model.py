"""Module 2 — 예측 모델 인터페이스.

소유자: P3  (이 파일은 P3만 수정한다)
계약: docs/INTERFACE.md  ★ predict() 시그니처는 P4·P5가 의존한다. 변경 금지.

입력 X: FEATURE_COLS 11개(원본 단위). grade 열은 선택(결측 대치에만 사용).
공정 변수 결측 정책
  A  공정 변수도 등급별 중앙값으로 대치 (Module 1 방식)
  B  공정 변수 NaN 유지 — NaN 을 직접 처리하는 boosting 전용
  C  공정 4변수가 모두 관측된 행만 학습, 공정 결측 입력은 예측 거절(NaN 반환)
성분 7개 결측은 세 정책 모두 fold 의 학습 데이터 등급별 중앙값으로 대치한다.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import FeatureBuilder, GradeMedianImputer

POLICIES = ("A", "B", "C")
NAN_CAPABLE = {"lgbm", "xgb", "hgb"}
MODELS = ("dummy", "ridge", "rf", "xgb", "lgbm", "hgb")

# 420행 규모에 맞춰 복잡도를 제한한 기본값. P2 노트북 설정과 같은 수준.
DEFAULT_PARAMS = {
    "rf": dict(n_estimators=400, min_samples_leaf=3, max_features=0.6),
    "lgbm": dict(n_estimators=300, learning_rate=0.04, num_leaves=7, min_child_samples=10,
                 reg_lambda=1.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
    "xgb": dict(n_estimators=300, learning_rate=0.04, max_depth=3, min_child_weight=3,
                reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8),
    "hgb": dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=7, min_samples_leaf=10,
                l2_regularization=1.0, early_stopping=False),
}


def make_estimator(name: str, params: dict | None = None):
    p = {**DEFAULT_PARAMS.get(name, {}), **(params or {})}
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()),
                         ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25)))])
    if name == "rf":
        return RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1, **p)
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=C.RANDOM_STATE, n_jobs=2, verbosity=-1, **p)
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=C.RANDOM_STATE, n_jobs=2, tree_method="hist",
                            objective="reg:squarederror", **p)
    if name == "hgb":
        return HistGradientBoostingRegressor(random_state=C.RANDOM_STATE, **p)
    raise ValueError(f"알 수 없는 모델: {name}")


class RecoveryModel:
    """Ni·Co 회수율 동시 예측 모델 (금속별 회귀 모델 2개)."""

    def __init__(self, model: str = "ridge", policy: str = "A", feature_set: str = "F0",
                 params: dict | None = None):
        if model not in MODELS or policy not in POLICIES or feature_set not in C.FEATURE_SETS:
            raise ValueError(f"잘못된 설정: {model}/{policy}/{feature_set}")
        if policy == "B" and model not in NAN_CAPABLE:
            raise ValueError(f"정책 B 는 NaN 처리 모델 전용: {model}")
        self.model = model
        self.policy = policy
        self.feature_set = feature_set
        self.params = dict(params or {})
        self.model_version = C.MODEL_VERSION

    @property
    def name(self) -> str:
        tag = f"{self.model}|{self.policy}|{self.feature_set}"
        if self.params:
            tag += "|" + ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return tag

    def _pipeline(self) -> Pipeline:
        impute_cols = C.COMPOSITION + (C.PROCESS if self.policy in ("A", "C") else [])
        return Pipeline([
            ("impute", GradeMedianImputer(columns=tuple(impute_cols))),
            ("features", FeatureBuilder(derived=tuple(C.FEATURE_SETS[self.feature_set]))),
            ("model", make_estimator(self.model, self.params)),
        ])

    def _inputs(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(C.FEATURE_COLS) - set(X.columns)
        if missing:
            raise ValueError(f"입력에 필요한 열이 없음: {sorted(missing)}")
        cols = (["grade"] if "grade" in X else []) + C.FEATURE_COLS
        out = X[cols].copy()
        out[C.FEATURE_COLS] = out[C.FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
        return out

    def accepts(self, X: pd.DataFrame) -> np.ndarray:
        """예측을 제공하는 행. 정책 C 는 공정 4변수가 모두 있는 행만."""
        if self.policy == "C":
            return X[C.PROCESS].notna().all(axis=1).to_numpy()
        return np.ones(len(X), dtype=bool)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "RecoveryModel":
        Xi = self._inputs(X)
        keep = self.accepts(Xi)
        if "grade" not in Xi:
            raise ValueError("학습 입력에는 grade 열이 필요하다 (등급별 대치)")
        self.pipelines_ = {
            t: self._pipeline().fit(Xi[keep], np.asarray(y[t], dtype=float)[keep]) for t in C.TARGETS
        }
        self.n_train_ = int(keep.sum())
        train = Xi[keep]
        self.feature_range_ = pd.DataFrame({
            "min": train[C.FEATURE_COLS].min(), "max": train[C.FEATURE_COLS].max(),
            "p01": train[C.FEATURE_COLS].quantile(0.01), "p99": train[C.FEATURE_COLS].quantile(0.99),
        })
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """shape (n, 2) 배열을 반환한다. 열 순서는 [ni_recovery_pct, co_recovery_pct].

        정책 C 에서 공정 변수가 빠진 행은 NaN (예측 거절).
        """
        Xi = self._inputs(X)
        out = np.full((len(Xi), 2), np.nan)
        ok = self.accepts(Xi)
        if ok.any():
            for j, t in enumerate(C.TARGETS):
                out[ok, j] = self.pipelines_[t].predict(Xi[ok])
        return out

    def predict_std(self, X: pd.DataFrame) -> np.ndarray:
        """참고용 오차 척도: 시간순 CV 공정 완전 검증군의 금속별 RMSE (행마다 동일).

        보정된 신뢰구간이 아니다. 학습 범위 밖·목표 부근에서 실제 오차는 더 클 수 있다.
        """
        std = getattr(self, "residual_rmse_", None)
        if std is None:
            raise AttributeError("residual_rmse_ 가 설정되지 않음 (train.py 로 저장한 모델만 제공)")
        return np.tile(np.asarray(std, dtype=float), (len(X), 1))
