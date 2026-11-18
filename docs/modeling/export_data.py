"""3D 씬이 읽는 data.json 을 파이프라인에서 재생성한다.

    python docs/modeling/export_data.py

선행 조건: artifacts/recovery_model.joblib 과 data/processed/batches.csv
(둘 다 .gitignore 대상이다. `docker compose up --build` 가 자동으로 만든다.)

씬은 브라우저 안에서만 도는 정적 페이지라 파이썬 모델을 호출할 수 없다.
그래서 선형 모델의 계수와 적용 범위 판정기의 학습점을 그대로 내보내
자바스크립트가 같은 계산을 재현하게 한다. 트리 앙상블은 옮길 수 없어 sigma 는
정확히 재현할 수 없으므로, 목표 달성 판정은 여기서 모듈 3 를 직접 호출해 구한
결과(rec)만 페이지에 싣는다. sigma 격자(sigma_surface)는 참고용으로만 내보낸다.
임의 조건의 정확한 판정은 Streamlit 앱(app/streamlit_app.py)을 쓴다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from module2_prediction import config as C  # noqa: E402
from module3_optimization.config import load_config  # noqa: E402
from module3_optimization.domain import build_domains  # noqa: E402

OUT = Path(__file__).with_name("data.json")
DECISION = ["acid_l", "temp_c", "time_hr"]
PER_GRADE = 18  # 등급당 표본 배치 수 (파일 크기와 다양성의 절충)


def ridge_spec(model, target: str) -> dict:
    """RecoveryModel 안의 Ridge 를 계수 수준으로 분해한다."""
    pipe = model.pipelines_[target]
    names = pipe.named_steps["features"].feature_names_
    inner = pipe.named_steps["model"]
    scaler, est = inner.steps[0][1], inner.steps[-1][1]
    if not hasattr(est, "coef_"):
        raise SystemExit(
            f"선형 모델이 아니라 계수를 내보낼 수 없다: {type(est).__name__}. "
            "정적 씬은 선형 모델에서만 성립한다 — Streamlit 앱을 쓴다."
        )
    return {
        "features": list(names),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": est.coef_.tolist(),
        "intercept": float(est.intercept_),
        "alpha": float(getattr(est, "alpha_", getattr(est, "alpha", 0.0))),
    }


def sigma_surface(model, feed: dict, temps, times, acid: float) -> dict:
    """중앙값 원료 기준 (온도 × 시간) 격자의 mu·sigma. 한 번에 예측한다."""
    rows = [{**feed, "acid_l": acid, "temp_c": t, "time_hr": h} for t in temps for h in times]
    X = pd.DataFrame(rows)
    mu = model.predict(X)[:, 0]
    sd = model.predict_std(X)[:, 0]
    return {
        "temp": [float(t) for t in temps],
        "time": [float(h) for h in times],
        "acid": float(acid),
        "mu": [round(float(v), 3) for v in mu],
        "sigma": [None if np.isnan(v) else round(float(v), 4) for v in sd],
    }


def sample_batches(df: pd.DataFrame) -> pd.DataFrame:
    """등급마다 고르게, 결정적으로 뽑는다 (난수 없음)."""
    picks = []
    for grade in ("A", "B", "C"):
        rows = df[df["grade"] == grade].sort_values("batch_id")
        if rows.empty:
            continue
        idx = np.linspace(0, len(rows) - 1, min(PER_GRADE, len(rows))).round().astype(int)
        picks.append(rows.iloc[np.unique(idx)])
    return pd.concat(picks).sort_values("batch_id")


def real_recommendations(blob: dict, df: pd.DataFrame, cfg: dict, doms) -> None:
    """배치별 추천을 **진짜 모듈 3 으로** 계산해 넣는다.

    브라우저에서 같은 탐색을 근사로 돌려 봤더니 판정이 뒤집혔다. 씬의 sigma 는
    (온도 × 시간) 격자를 황산 중앙값으로 고정해 만든 것인데, 비용 최소해는
    황산 상한 쪽으로 몰리고 그 구간에서 실제 sigma 가 최대 1.7 더 크다.
    그래서 근사는 '달성 가능' 이라 답하고 진짜 모델은 'infeasible' 이라 답했다.

    정적 페이지가 실제 시스템과 다른 답을 내면 안 되므로, 추천과 가부 판정은
    여기서 파이썬으로 계산해 넣고 브라우저는 그 결과를 보여주기만 한다.
    슬라이더를 움직였을 때의 즉석 표시만 근사로 남는다(페이지에 명시).
    """
    from module3_optimization.problem import InputError
    from module3_optimization.recommend import feed_from_row, recommend

    process_dom, feed_dom = doms
    rec, n_ok = {}, 0
    for b in blob["batches"]:
        row = df[df["batch_id"] == b["batch_id"]].iloc[0]
        try:
            r = recommend(feed_from_row(row), cfg=cfg, algo="grid",
                          domain=process_dom, feed_domain=feed_dom)
        except InputError as exc:
            rec[b["batch_id"]] = {"ok": 0, "error": str(exc)}
            continue
        cell = {"ok": 1 if r.status == "ok" else 0}
        if r.recipe:
            cell |= {"a": int(r.recipe["acid_l"]), "t": int(r.recipe["temp_c"]),
                     "h": float(r.recipe["time_hr"])}
        if r.status != "ok" and r.shortfall:
            cell["gap"] = round(max(float(v) for v in r.shortfall.values()), 2)
        if r.prediction:
            cell["ni"] = r.prediction["ni"]["mean"]
            cell["co"] = r.prediction["co"]["mean"]
        rec[b["batch_id"]] = cell
        n_ok += cell["ok"]

    blob["rec"] = rec
    first = next((b["batch_id"] for b in blob["batches"] if rec[b["batch_id"]]["ok"]), None)
    blob["opening"] = {"batch_id": first, **rec[first]} if first else None
    print(f"  모듈 3 실제 추천: 달성 가능 {n_ok}/{len(blob['batches'])}건, 시작 배치 {first}")


def main() -> int:
    model_path = ROOT / "artifacts/recovery_model.joblib"
    proc_path = ROOT / "data/processed/batches.csv"
    for p in (model_path, proc_path):
        if not p.exists():
            raise SystemExit(
                f"{p.relative_to(ROOT)} 가 없다. 먼저 파이프라인을 돌린다:\n"
                "  python -m module1_preprocessing.cleaning\n"
                "  python -m module2_prediction.train"
            )

    model = joblib.load(model_path)
    df = pd.read_csv(proc_path)
    cfg = load_config()
    doms = build_domains(cfg)
    process_dom = doms[0]
    print(f"모델 {model.model} | {model.policy} | {model.feature_set}  학습 {model.n_train_}행")

    train = df.dropna(subset=DECISION)[DECISION]
    blob: dict = {
        "model": {
            "name": f"{model.model}|{model.policy}|{model.feature_set}",
            "version": model.model_version,
            "residual_rmse": [float(v) for v in model.residual_rmse_],
        },
        "ridge": {t: ridge_spec(model, t) for t in C.TARGETS},
        "domain": {
            "cols": DECISION,
            "k": int(process_dom.k),
            "percentile": float(process_dom.percentile),
            "threshold": float(process_dom.threshold),
            "mean": process_dom.scaler.mean_.tolist(),
            "scale": process_dom.scaler.scale_.tolist(),
            "train": [[round(float(v), 2) for v in row] for row in train.to_numpy()],
        },
        "targets": {
            "ni": float(cfg["targets"]["ni_recovery_pct"]),
            "co": float(cfg["targets"]["co_recovery_pct"]),
            "z": float(cfg["uncertainty"]["chance_z"]),
        },
        "ratio": {
            "min": float(cfg["ratio_constraints"]["acid_per_feed"]["min"]),
            "max": float(cfg["ratio_constraints"]["acid_per_feed"]["max"]),
        },
    }

    keep = ["batch_id", "grade", *C.COMPOSITION, "feed_kg",
            "acid_l", "temp_c", "time_hr", *C.TARGETS]
    keep = [c for c in keep if c in df.columns]
    sample = sample_batches(df)[keep]
    blob["batches"] = json.loads(sample.round(4).to_json(orient="records"))
    print(f"  표본 배치 {len(sample)}건 (등급별 {PER_GRADE}건 목표)")

    med = df[list(C.COMPOSITION) + ["feed_kg"]].median()
    blob["sigma_surface"] = sigma_surface(
        model, med.to_dict(),
        np.arange(40, 96, 5.0), np.arange(1, 8.01, 0.5),
        float(df["acid_l"].median()),
    )
    sg = [v for v in blob["sigma_surface"]["sigma"] if v is not None]
    print(f"  sigma 격자 {len(blob['sigma_surface']['mu'])}점  범위 {min(sg):.3f}–{max(sg):.3f}")

    print("  모듈 3 를 배치마다 호출하는 중… (1건당 1초 안팎)")
    real_recommendations(blob, df, cfg, doms)

    OUT.write_text(json.dumps(blob, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"저장 → {OUT.relative_to(ROOT)}  {OUT.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
