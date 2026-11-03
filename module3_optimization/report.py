"""결과 출력 (텍스트 카드 · JSON · CSV).

소유자: P4  (이 파일은 P4만 수정한다)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from module3_optimization.problem import STATUS_INFEASIBLE, STATUS_OK, Recipe

CONFIDENCE_KO = {"high": "높음", "medium": "보통", "low": "낮음", None: "-"}


def _header(recipe: Recipe, title: str) -> str:
    batch = recipe.batch_id or "직접 입력"
    return f"[{title}]  배치 {batch}"


def _dev_notice(recipe: Recipe) -> list[str]:
    version = str(recipe.meta.get("model_version", ""))
    if version.startswith(("mock", "proxy")):
        return [f"  ※ 개발·검증용 모델({version}) 결과 — 실제 추천 수치가 아님"]
    return []


def _pred_line(recipe: Recipe) -> str:
    ni, co = recipe.prediction["ni"], recipe.prediction["co"]
    return (
        f"  예상 Ni {ni['mean']:.1f}% (90% 구간 {ni['low']:.1f}~{ni['high']:.1f})"
        f" · Co {co['mean']:.1f}% ({co['low']:.1f}~{co['high']:.1f})"
    )


def _recipe_line(recipe: Recipe) -> str:
    r = recipe.recipe
    return (
        f"  황산 {r['acid_l']:g} L ({r['acid_per_feed']:.2f} L/kg)"
        f" · {r['temp_c']:g} ℃ · {r['time_hr']:g} 시간"
    )


def format_card(recipe: Recipe) -> str:
    """작업자용 레시피 카드. infeasible / out_of_domain 이면 안내문."""
    t = recipe.target
    goal = f"  목표 Ni ≥ {t['ni_recovery_pct']:g}% · Co ≥ {t['co_recovery_pct']:g}%"

    if recipe.status == STATUS_OK:
        similar = (
            f" — 유사 과거 배치 {recipe.similar_batches}건"
            if recipe.similar_batches is not None
            else ""
        )
        lines = [
            _header(recipe, "추천 레시피"),
            goal,
            _recipe_line(recipe),
            _pred_line(recipe),
            f"  신뢰도: {CONFIDENCE_KO[recipe.confidence]}{similar}",
        ]
    elif recipe.status == STATUS_INFEASIBLE:
        lines = [_header(recipe, "목표 달성 불가"), goal]
        if recipe.recipe is not None:
            lines += ["  가장 근접한 조건:", _recipe_line(recipe), _pred_line(recipe)]
        lines.append(f"  {recipe.message}")
    else:
        lines = [
            _header(recipe, "추천 불가 — 학습 범위 밖 원료"),
            f"  {recipe.message}",
        ]

    return "\n".join(lines + _dev_notice(recipe))


def write_json(recipe: Recipe, path: str | Path) -> None:
    Path(path).write_text(recipe.to_json(), encoding="utf-8")


def to_frame(recipes: list[Recipe]) -> pd.DataFrame:
    """여러 결과를 한 행씩 평탄화."""
    rows = []
    for r in recipes:
        row = {"batch_id": r.batch_id, "status": r.status}
        row.update(r.recipe or {})
        for metal, stats in (r.prediction or {}).items():
            row.update({f"{metal}_{k}": v for k, v in stats.items()})
        for metal, value in (r.shortfall or {}).items():
            row[f"shortfall_{metal}"] = value
        row.update(
            confidence=r.confidence,
            similar_batches=r.similar_batches,
            domain_distance=r.domain_distance,
            algorithm=r.meta.get("algorithm"),
            model_version=r.meta.get("model_version"),
            message=r.message,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(recipes: list[Recipe], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: 엑셀에서 한글이 깨지지 않게
    to_frame(recipes).to_csv(path, index=False, encoding="utf-8-sig")
