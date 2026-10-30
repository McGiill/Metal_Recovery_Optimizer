"""문제 정의 (Problem) 와 출력 구조 (Recipe).

소유자: P4  (이 파일은 P4만 수정한다)

Problem 은 알고리즘과 무관하다. 모든 최적화기는 Problem.evaluate() 만 호출한다.

    minimize   cost(u)                      (기본: acid_l)
    subject to μ_Ni − z·σ_Ni ≥ 95,  μ_Co − z·σ_Co ≥ 90
               u ∈ 박스 · 비율 · (적용 범위)
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from module3_optimization import constraints as C
from module3_optimization.config import get, load_config
from module3_optimization.predictors import PRED_COLS, Predictor, build_predictor

if TYPE_CHECKING:
    from module3_optimization.domain import DomainChecker

DECISION_VARS = ("acid_l", "temp_c", "time_hr")
FEED_COLS = (
    "ni_pct",
    "co_pct",
    "li_pct",
    "mn_pct",
    "cu_pct",
    "fe_pct",
    "al_pct",
    "feed_kg",
)
OPTIONAL_FEED_KEYS = ("batch_id", "grade")
TARGET_KEYS = ("ni_recovery_pct", "co_recovery_pct")

STATUS_OK = "ok"
STATUS_INFEASIBLE = "infeasible"
STATUS_OUT_OF_DOMAIN = "out_of_domain"
STATUSES = (STATUS_OK, STATUS_INFEASIBLE, STATUS_OUT_OF_DOMAIN)
SCHEMA_VERSION = "1.0"


class InputError(ValueError):
    """원료 입력이 비었거나 숫자가 아닐 때. API 에서는 422 로 매핑한다."""

    def __init__(
        self,
        message: str,
        missing: list[str] | None = None,
        invalid: list[str] | None = None,
    ):
        super().__init__(message)
        self.missing = missing or []
        self.invalid = invalid or []


def validate_feed(feed: dict) -> dict:
    """필수 원료 키를 확인하고 float 로 바꾼다. 결측은 대치하지 않고 거절한다."""
    missing = [c for c in FEED_COLS if feed.get(c) is None]
    invalid = []
    clean: dict = {}
    for col in FEED_COLS:
        if col in missing:
            continue
        try:
            value = float(feed[col])
        except (TypeError, ValueError):
            invalid.append(col)
            continue
        if not math.isfinite(value):
            invalid.append(col)
        else:
            clean[col] = value

    if missing or invalid:
        parts = []
        if missing:
            parts.append(f"누락: {', '.join(missing)}")
        if invalid:
            parts.append(f"숫자 아님/NaN: {', '.join(invalid)}")
        raise InputError(
            "원료 입력 오류 — " + " / ".join(parts), missing=missing, invalid=invalid
        )
    if clean["feed_kg"] <= 0:
        raise InputError(
            "원료 입력 오류 — feed_kg 는 0보다 커야 한다.", invalid=["feed_kg"]
        )

    for key in OPTIONAL_FEED_KEYS:
        if feed.get(key) is not None:
            clean[key] = feed[key]
    return clean


def identity_features(X: pd.DataFrame) -> pd.DataFrame:
    """P2 add_features 가 오기 전까지 쓰는 자리 표시."""
    return X


def build_feature_fn(cfg: dict) -> Callable[[pd.DataFrame], pd.DataFrame]:
    kind = get(cfg, "features.type", "identity")
    if kind == "identity":
        return identity_features
    if kind == "p2":
        from module1_preprocessing.features import add_features

        return add_features
    raise ValueError(f"알 수 없는 features.type: {kind}")


@dataclass
class Problem:
    feed: dict
    cfg: dict
    predictor: Predictor
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame] = identity_features
    domain: DomainChecker | None = None
    target: dict | None = None  # None → cfg.targets, 일부만 주면 그 키만 덮어쓴다

    def __post_init__(self):
        self.feed = validate_feed(self.feed)
        self.target = {**self.cfg["targets"], **(self.target or {})}

    @classmethod
    def from_config(
        cls,
        feed: dict,
        cfg: dict | None = None,
        target: dict | None = None,
        domain: DomainChecker | None = None,
    ) -> Problem:
        cfg = cfg or load_config()
        return cls(
            feed=feed,
            cfg=cfg,
            predictor=build_predictor(cfg),
            feature_fn=build_feature_fn(cfg),
            domain=domain,
            target=target,
        )

    @property
    def variables(self) -> dict:
        return self.cfg["decision_variables"]

    @property
    def model_version(self) -> str:
        return getattr(self.predictor, "name", type(self.predictor).__name__)

    # ── 평가 ────────────────────────────────────────────────

    def evaluate(self, U: pd.DataFrame) -> pd.DataFrame:
        """후보 공정 조건 여러 개 → 예측 · 판정값 · 제약 · 비용이 붙은 DataFrame.

        U 는 DECISION_VARS 컬럼을 가진 여러 행. 예측은 한 번만 호출한다 (배치).
        반환 index 는 U 의 index 와 같다.
        """
        missing = [v for v in DECISION_VARS if v not in U]
        if missing:
            raise KeyError(f"결정 변수 컬럼 누락: {missing}")
        U = U[list(DECISION_VARS)].astype(float)

        X = self.feature_fn(self._attach_feed(U))
        pred = self.predictor.predict(X)
        if len(pred) != len(U):
            raise ValueError(
                f"예측 행 수 {len(pred)} 가 입력 행 수 {len(U)} 와 다르다."
            )
        # feature_fn(P2)이 index 를 바꿔도 concat 에서 행이 어긋나지 않게 위치 기준으로 맞춘다.
        X = X.set_axis(U.index)
        pred = pred.set_axis(U.index)

        out = pd.concat([U, pred[PRED_COLS]], axis=1)
        out["acid_per_feed"] = out["acid_l"] / self.feed["feed_kg"]
        out["uncertainty"] = out["ni_std"] + out["co_std"]

        z = (
            get(self.cfg, "uncertainty.chance_z", 0.0)
            if get(self.cfg, "uncertainty.use_chance_constraint")
            else 0.0
        )
        out["ni_judged"] = out["ni_pred"] - z * out["ni_std"]
        out["co_judged"] = out["co_pred"] - z * out["co_std"]
        out["shortfall_ni"] = (self.target["ni_recovery_pct"] - out["ni_judged"]).clip(
            lower=0
        )
        out["shortfall_co"] = (self.target["co_recovery_pct"] - out["co_judged"]).clip(
            lower=0
        )
        out["shortfall_total"] = out["shortfall_ni"] + out["shortfall_co"]

        if self.domain is not None:
            out["domain_distance"] = self.domain.distance(X)

        results = self._constraints(out)
        hard = [name for name in results if not name.startswith("target_")]
        for name, (ok, viol) in results.items():
            out[f"ok_{name}"] = ok
            out[f"viol_{name}"] = viol
        out["hard_ok"] = out[[f"ok_{n}" for n in hard]].all(axis=1)
        out["feasible"] = out[[f"ok_{n}" for n in results]].all(axis=1)
        out["violation"] = out[[f"viol_{n}" for n in results]].sum(axis=1)
        out["cost"] = self.cost(out)
        return out

    def cost(self, df: pd.DataFrame) -> pd.Series:
        weights = get(self.cfg, "cost.weights", {"acid_l": 1.0})
        total = pd.Series(0.0, index=df.index)
        for var, w in weights.items():
            if w:
                total += w * df[var]
        return total

    def _attach_feed(self, U: pd.DataFrame) -> pd.DataFrame:
        X = U.copy()
        for col, value in self.feed.items():
            if col != "batch_id":
                X[col] = value
        return X

    def _constraints(self, out: pd.DataFrame) -> dict[str, C.ConstraintResult]:
        results: dict[str, C.ConstraintResult] = {}
        for var, spec in self.variables.items():
            results[f"box_{var}"] = C.box(out, var, spec["min"], spec["max"])

        for col, spec in (self.cfg.get("ratio_constraints") or {}).items():
            spec = spec or {}
            if spec.get("min") is not None or spec.get("max") is not None:
                results[f"ratio_{col}"] = C.ratio(
                    out, col, spec.get("min"), spec.get("max")
                )

        if self.domain is not None:
            results["domain"] = C.bounded(
                out["domain_distance"], hi=self.domain.threshold
            )

        results["target_ni"] = C.target(
            out, "ni_judged", self.target["ni_recovery_pct"]
        )
        results["target_co"] = C.target(
            out, "co_judged", self.target["co_recovery_pct"]
        )
        return results


# ── 출력 구조 ──────────────────────────────────────────────


@dataclass
class Recipe:
    """추천 결과. to_dict() 가 P5 로 넘어가는 JSON 스키마다 (README 4.3)."""

    status: str
    target: dict
    batch_id: str | None = None
    recipe: dict | None = None  # acid_l, temp_c, time_hr, acid_per_feed
    prediction: dict | None = None  # {"ni": {mean, low, high}, "co": {...}}
    shortfall: dict | None = None  # infeasible 일 때 {"ni": %p, "co": %p}
    confidence: str | None = None  # high | medium | low
    similar_batches: int | None = None
    domain_distance: float | None = None
    message: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status 는 {STATUSES} 중 하나여야 한다: {self.status}")

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "batch_id": self.batch_id,
            "target": self.target,
            "recipe": self.recipe,
            "prediction": self.prediction,
            "shortfall": self.shortfall,
            "confidence": self.confidence,
            "similar_batches": self.similar_batches,
            "domain_distance": self.domain_distance,
            "message": self.message,
            "meta": self.meta,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
