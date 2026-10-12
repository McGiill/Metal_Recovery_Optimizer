"""추천 파이프라인 + CLI.

소유자: P4  (이 파일은 P4만 수정한다)

흐름:
  1. validate_feed (결측이면 InputError — 대치하지 않는다)
  2. 원료가 학습 범위 밖 → status = out_of_domain
  3. 최적화 (기본 grid)
  4. 실행 가능 해 → 현장 단위 반올림 · 재평가 → 여전히 실행 가능한 첫 후보 → ok
     없으면 select_closest → infeasible + shortfall
  5. 예측 구간 μ ± interval_z·σ, 신뢰도 등급, 유사 배치 수
  6. Recipe 반환

    python -m module3_optimization.recommend --batch-id BP-2025-0048
    python -m module3_optimization.recommend --feed feed.json --ni-min 95 --co-min 90
    python -m module3_optimization.recommend --all --out artifacts/recipes.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime

import pandas as pd

from module3_optimization.config import config_hash, get, load_config
from module3_optimization.constraints import round_to_unit
from module3_optimization.domain import DomainChecker, build_domains
from module3_optimization.optimizers import get_optimizer, select_closest
from module3_optimization.optimizers.base import rank_feasible
from module3_optimization.predictors import build_predictor
from module3_optimization.problem import (
    DECISION_VARS,
    FEED_COLS,
    STATUS_INFEASIBLE,
    STATUS_OK,
    STATUS_OUT_OF_DOMAIN,
    InputError,
    Problem,
    Recipe,
    build_feature_fn,
    validate_feed,
)

# 반올림 후 재검증할 상위 후보 수. Grid 는 격자가 현장 단위라 첫 후보가 그대로 통과한다.
SNAP_CANDIDATES = 50
BLEND_HINT_IMPURITY = 6.0  # 불순물 합이 이보다 크면 블렌딩 제안


def recommend(
    feed: dict,
    target: dict | None = None,
    algo: str = "grid",
    cfg: dict | None = None,
    domain: DomainChecker | None = None,
    feed_domain: DomainChecker | None = None,
    seed: int = 0,
    budget: int | None = None,
) -> Recipe:
    """원료 하나에 대한 추천. domain · feed_domain 을 안 주면 config 에 따라 만든다."""
    start = time.perf_counter()
    cfg = cfg or load_config()
    clean = validate_feed(feed)
    target_full = {**cfg["targets"], **(target or {})}

    if domain is None and feed_domain is None and get(cfg, "domain.enabled"):
        domain, feed_domain = build_domains(cfg)

    predictor = build_predictor(cfg)
    meta = {
        "algorithm": algo,
        "n_evals": 0,
        "elapsed_ms": 0,
        "model_version": getattr(predictor, "name", type(predictor).__name__),
        "config_hash": config_hash(cfg),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    def finish(**kwargs) -> Recipe:
        meta["elapsed_ms"] = round((time.perf_counter() - start) * 1000)
        return Recipe(
            target=target_full, batch_id=clean.get("batch_id"), meta=meta, **kwargs
        )

    # 2. 원료 자체가 범위 밖
    if feed_domain is not None:
        dist = float(feed_domain.distance(pd.DataFrame([clean]))[0])
        if dist > feed_domain.threshold:
            return finish(
                status=STATUS_OUT_OF_DOMAIN,
                domain_distance=round(dist, 3),
                message=(
                    "원료 성분이 과거 데이터와 많이 달라 예측을 신뢰할 수 없어 추천하지 않습니다 "
                    f"(거리 {dist:.2f} / 기준 {feed_domain.threshold:.2f}). "
                    "성분 재분석 또는 소규모 시험 후 데이터 추가를 권장합니다."
                ),
            )

    # 3. 최적화
    problem = Problem(
        feed=feed,
        cfg=cfg,
        predictor=predictor,
        feature_fn=build_feature_fn(cfg),
        domain=domain,
        target=target,
    )
    result = get_optimizer(algo).run(problem, budget=budget, seed=seed)
    meta["algorithm"] = result.algorithm
    meta["n_evals"] = result.n_evals

    # 4. 실행 가능 해 → 반올림 · 재검증
    row = _snap_and_verify(problem, result.history)
    if row is not None:
        status = STATUS_OK
    else:
        status = STATUS_INFEASIBLE
        row = select_closest(result.history)
        if row is None:
            return finish(
                status=STATUS_INFEASIBLE,
                message="박스 · 비율 · 적용 범위 제약을 모두 만족하는 공정 조건이 없습니다.",
            )

    # 5. 예측 구간 · 신뢰도
    z = get(cfg, "uncertainty.interval_z", 1.645)
    similar = None
    if domain is not None:
        similar = int(domain.similar_count(_feature_row(problem, row))[0])
    has_std = bool(row["ni_std"] > 0 or row["co_std"] > 0)

    recipe = {var: _num(row[var]) for var in DECISION_VARS}
    recipe["acid_per_feed"] = round(float(row["acid_per_feed"]), 2)
    prediction = {
        metal: _interval(row[f"{metal}_pred"], row[f"{metal}_std"], z)
        for metal in ("ni", "co")
    }
    shortfall = None
    if status == STATUS_INFEASIBLE:
        shortfall = {
            "ni": round(float(row["shortfall_ni"]), 2),
            "co": round(float(row["shortfall_co"]), 2),
        }

    return finish(
        status=status,
        recipe=recipe,
        prediction=prediction,
        shortfall=shortfall,
        confidence=grade_confidence(cfg, status, similar, has_std),
        similar_batches=similar,
        domain_distance=(
            round(float(row["domain_distance"]), 3)
            if "domain_distance" in row
            else None
        ),
        message=_message(status, clean, row, shortfall),
    )


def _snap_and_verify(problem: Problem, history: pd.DataFrame) -> pd.Series | None:
    """상위 실행 가능 후보를 현장 단위로 반올림 → 재평가 → 여전히 실행 가능한 최선."""
    ranked = rank_feasible(history, problem.cfg["tie_break"]).head(SNAP_CANDIDATES)
    if ranked.empty:
        return None
    snapped = round_to_unit(ranked[list(DECISION_VARS)], problem.variables)
    snapped = snapped.drop_duplicates()
    rechecked = problem.evaluate(snapped)
    survivors = rank_feasible(rechecked, problem.cfg["tie_break"])
    return None if survivors.empty else survivors.iloc[0]


def _interval(mean: float, std: float, z: float) -> dict:
    """μ ± z·σ, 회수율이므로 [0, 100] 으로 자른다."""
    clip = lambda v: round(min(max(float(v), 0.0), 100.0), 2)
    return {
        "mean": clip(mean),
        "low": clip(mean - z * std),
        "high": clip(mean + z * std),
    }


def _feature_row(problem: Problem, row: pd.Series) -> pd.DataFrame:
    U = pd.DataFrame([{var: row[var] for var in DECISION_VARS}])
    return problem.feature_fn(problem._attach_feed(U))


def grade_confidence(
    cfg: dict,
    status: str,
    similar: int | None,
    has_std: bool,
) -> str:
    """high / medium / low. 임계값은 config.yaml confidence.

    목표 여유(margin)는 쓰지 않는다. 비용 최소해는 정의상 목표 경계 위에 있어 여유가
    항상 0 에 가깝고, 안전 여유는 이미 확률 제약(μ − 1.28σ)에 들어 있다.
    """
    if status != STATUS_OK:
        return "low"
    if similar is None:  # 적용 범위 판정기 없음 → 근거 부족
        return "medium"
    if has_std and similar >= get(cfg, "confidence.high_similar", 5):
        return "high"
    if similar >= get(cfg, "confidence.medium_similar", 2):
        return "medium"
    return "low"


def _message(status: str, feed: dict, row: pd.Series, shortfall: dict | None) -> str:
    if status == STATUS_OK:
        return "목표 달성 가능 (확률 제약 기준)."
    impurity = feed["cu_pct"] + feed["fe_pct"] + feed["al_pct"]
    text = (
        f"허용 범위 안에서는 목표 달성이 어렵습니다. 가장 근접한 조건의 예상 회수율은 "
        f"Ni {row['ni_pred']:.1f}% / Co {row['co_pred']:.1f}% 입니다 "
        f"(부족분 Ni {shortfall['ni']:.1f}%p, Co {shortfall['co']:.1f}%p, 확률 제약 기준)."
    )
    if impurity > BLEND_HINT_IMPURITY:
        text += f" 불순물 합 {impurity:.1f}% — 저불순물 원료와의 블렌딩을 검토하세요."
    return text


def _num(value) -> float | int:
    value = float(value)
    return int(value) if value.is_integer() else round(value, 3)


# ── 배치 ID · 일괄 ─────────────────────────────────────


def feed_from_row(row: pd.Series) -> dict:
    feed = {col: row[col] for col in FEED_COLS}
    for key in ("batch_id", "grade"):
        if key in row and pd.notna(row[key]):
            feed[key] = row[key]
    return {
        k: (None if (not isinstance(v, str) and pd.isna(v)) else v)
        for k, v in feed.items()
    }


def recommend_all(
    data: pd.DataFrame, cfg: dict | None = None, algo: str = "grid", **kwargs
) -> tuple[list[Recipe], list[tuple[str, str]]]:
    """여러 배치 일괄 추천. 판정기는 한 번만 만든다. (recipes, [(batch_id, 오류)]) 반환."""
    cfg = cfg or load_config()
    domain = feed_domain = None
    if get(cfg, "domain.enabled"):
        domain, feed_domain = build_domains(cfg)
    recipes, errors = [], []
    for _, row in data.iterrows():
        try:
            recipes.append(
                recommend(
                    feed_from_row(row),
                    cfg=cfg,
                    algo=algo,
                    domain=domain,
                    feed_domain=feed_domain,
                    **kwargs,
                )
            )
        except InputError as exc:
            errors.append((str(row.get("batch_id")), str(exc)))
    return recipes, errors


def main(argv: list[str] | None = None) -> int:
    from module3_optimization import report

    parser = argparse.ArgumentParser(description="Module 3 — 최적 공정 레시피 추천")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--feed", help="원료 성분 JSON 파일")
    src.add_argument("--batch-id", help="원본 데이터의 배치 ID")
    src.add_argument("--all", action="store_true", help="원본 데이터 전체 배치")
    parser.add_argument("--algo", default="grid")
    parser.add_argument("--ni-min", type=float)
    parser.add_argument("--co-min", type=float)
    parser.add_argument("--json", action="store_true", help="카드 대신 JSON 출력")
    parser.add_argument("--out", help="--all 결과 CSV 경로")
    parser.add_argument(
        "--no-domain", action="store_true", help="적용 범위 필터 끄기 (비교용)"
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.no_domain:
        cfg["domain"]["enabled"] = False
    target = {}
    if args.ni_min is not None:
        target["ni_recovery_pct"] = args.ni_min
    if args.co_min is not None:
        target["co_recovery_pct"] = args.co_min
    if target:
        cfg["targets"] = {**cfg["targets"], **target}

    if args.all or args.batch_id:
        from module3_optimization.analysis.load_raw import load_minimal

        data = load_minimal()
        if args.batch_id:
            data = data[data["batch_id"] == args.batch_id]
            if data.empty:
                print(f"배치 ID 없음: {args.batch_id}", file=sys.stderr)
                return 2

    try:
        if args.all:
            recipes, errors = recommend_all(data, cfg=cfg, algo=args.algo)
            out = args.out or "artifacts/recipes.csv"
            report.write_csv(recipes, out)
            counts = pd.Series([r.status for r in recipes]).value_counts().to_dict()
            print(
                f"{len(recipes)}건 저장 → {out}  {counts}  (입력 결측으로 건너뜀 {len(errors)}건)"
            )
            return 0
        if args.batch_id:
            feed = feed_from_row(data.iloc[0])
        else:
            with open(args.feed, encoding="utf-8") as f:
                feed = json.load(f)
        recipe = recommend(feed, cfg=cfg, algo=args.algo)
    except InputError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(recipe.to_json() if args.json else report.format_card(recipe))
    return 0


if __name__ == "__main__":
    sys.exit(main())
