"""Module 2 — 학습 · 튜닝 · 평가.

소유자: P3  (이 파일은 P3만 수정한다)

실행 (저장소 루트):
    python -m module1_preprocessing.cleaning       # data/processed/batches.csv
    python -m module2_prediction.train

절차 (docs: Module2 개발계획)
  1. 같은 LightGBM 으로 공정 결측 정책 A/B/C × F0/F1 비교 (+ Module 1 전역 대치값 참고)
  2. 모델별(Ridge/RF/XGBoost/LightGBM) 특징 집합 F0~F4 비교
  3. 단순한 모델부터 비교해 최종 모델 선택 → 제한된 튜닝
  4. 개발 구간 전체로 재학습 → 시험 구간 1회 평가, 하위군·목표 영역·중요도·학습곡선
  5. 전체 420행으로 재학습한 모델을 artifacts/ 에 저장 (Module 3 predictor.type: real)
선택은 모두 개발 구간의 시간순 CV 로만 한다. 시험 구간은 선택에 쓰지 않는다.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data import check_against_raw, load_module1, split_dev_test
from .evaluation import (forward_cv, metrics, paired_better, predict_frame, subgroup_table,
                         summarize)
from .model import NAN_CAPABLE, RecoveryModel

TUNE_GRID = {
    "lgbm": dict(learning_rate=[0.02, 0.04, 0.08], num_leaves=[4, 7, 15], min_child_samples=[5, 10, 20],
                 reg_lambda=[0.0, 1.0, 5.0], n_estimators=[200, 400, 800]),
    "xgb": dict(learning_rate=[0.02, 0.04, 0.08], max_depth=[2, 3, 4], min_child_weight=[1, 3, 6],
                reg_lambda=[0.5, 1.0, 5.0], n_estimators=[200, 400, 800]),
    "rf": dict(min_samples_leaf=[1, 3, 5, 10], max_features=[0.33, 0.6, 1.0], max_depth=[None, 6, 10]),
}
N_TUNE = 12


# ── 팀 뼈대 인터페이스 ─────────────────────────────────────────
def split(df: pd.DataFrame):
    """analysis_date(=배치ID) 기준 시간순으로 개발/시험을 분할한다. 검증은 개발 내부 전방 CV."""
    return split_dev_test(df)


def train_model(train: pd.DataFrame, **spec) -> RecoveryModel:
    return RecoveryModel(**spec).fit(train, train[C.TARGETS])


def evaluate(model: RecoveryModel, test: pd.DataFrame) -> dict:
    """{'rmse_ni':…, 'mae_ni':…, 'r2_ni':…, 'rmse_co':…, …} 를 반환한다."""
    return metrics(predict_frame(model, test))


# ── 실험 ───────────────────────────────────────────────────────
class Runner:
    def __init__(self, dev: pd.DataFrame):
        self.dev = dev
        self.results: dict[str, dict] = {}
        self.oof: dict[str, pd.DataFrame] = {}

    def run(self, stage: str, m1_filled=False, **spec) -> dict:
        probe = RecoveryModel(**spec)
        name = probe.name + ("|m1_filled" if m1_filled else "")
        if name not in self.results:
            oof = forward_cv(lambda: RecoveryModel(**spec), self.dev, m1_filled=m1_filled)
            res = summarize(oof, name)
            res.update(stage=stage, spec_args=spec, m1_filled=m1_filled)
            self.results[name], self.oof[name] = res, oof
            print(f"  [{stage}] {name:<55} CV RMSE {res['cv_rmse_mean']:.3f} ± {res['cv_rmse_std']:.3f}"
                  f"  coverage(all) {res['all_rows.coverage']:.2f}")
        return self.results[name]

    def table(self) -> pd.DataFrame:
        rows = []
        for r in self.results.values():
            row = {k: v for k, v in r.items() if k not in ("fold_rmse", "spec_args")}
            row.update({f"fold{i}": v for i, v in enumerate(r["fold_rmse"])})
            rows.append(row)
        return pd.DataFrame(rows)


def choose(base: dict, candidates: list[dict]) -> dict:
    """base 대비 paired_better 인 후보 중 평균 RMSE 최저. 없으면 base."""
    ok = [c for c in candidates if paired_better(c, base)]
    return min(ok, key=lambda r: r["cv_rmse_mean"]) if ok else base


def sample_grid(grid: dict, n: int, seed: int = C.RANDOM_STATE) -> list[dict]:
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    while len(out) < n:
        p = {k: v[rng.integers(len(v))] for k, v in grid.items()}
        key = tuple(sorted(p.items(), key=lambda kv: kv[0]))
        if key not in seen:
            seen.add(key)
            out.append({k: (None if v is None else (int(v) if isinstance(v, (int, np.integer)) else
                            float(v) if isinstance(v, (float, np.floating)) else v)) for k, v in p.items()})
    return out


def main() -> None:
    t0 = time.time()
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_module1()
    check_against_raw(df)
    dev, test = split(df)
    info = {
        "rows": len(df), "dev_rows": len(dev), "test_rows": len(test),
        "dev_process_complete": int(dev["process_complete"].sum()),
        "test_process_complete": int(test["process_complete"].sum()),
        "test_first_batch": test["batch_id"].iloc[0], "test_start_date": C.TEST_START_DATE,
    }
    print("데이터:", info)
    R = Runner(dev)

    # 1. 공정 결측 정책 (같은 LightGBM)
    print("\n[1] 공정 결측 정책 A/B/C × F0/F1 (LightGBM 고정)")
    pol = {(p, f): R.run("1_policy", model="lgbm", policy=p, feature_set=f)
           for p in "ABC" for f in ("F0", "F1")}
    R.run("1_policy_ref", m1_filled=True, model="lgbm", policy="A", feature_set="F0")
    qualified = [p for p in "BC" if all(paired_better(pol[(p, f)], pol[("A", f)]) for f in ("F0", "F1"))]
    policy = min(qualified, key=lambda p: pol[(p, "F0")]["cv_rmse_mean"]) if qualified else "A"
    print(f"  → 선택 정책: {policy} (A 대비 F0·F1 모두에서 3/4 fold 이상 개선한 정책: {qualified or '없음'})")

    # 2. 모델별 정책 × 특징 집합 (기준: 같은 모델의 A·F0)
    print("\n[2] 모델별 정책 × 특징 집합 F0~F4")
    best_by_model = {"dummy": R.run("2_features", model="dummy", policy="A", feature_set="F0")}
    for m in ("ridge", "rf", "xgb", "lgbm"):
        policies = "ABC" if m in NAN_CAPABLE else "AC"
        res = {(p, f): R.run("2_features", model=m, policy=p, feature_set=f)
               for p in policies for f in C.FEATURE_SETS}
        best_by_model[m] = choose(res[("A", "F0")], [r for k, r in res.items() if k != ("A", "F0")])
        print(f"  → {m}: {best_by_model[m]['spec']}")

    # 3. 단순한 모델부터 비교
    print("\n[3] 모델 선택 (Ridge → RF → XGBoost → LightGBM 순, 복잡한 모델은 3/4 fold 이상 개선 시 채택)")
    final = best_by_model["ridge"]
    for m in ("rf", "xgb", "lgbm"):
        if paired_better(best_by_model[m], final):
            final = best_by_model[m]
    print(f"  → 최종 후보: {final['spec']}")

    # 4. 제한된 튜닝
    spec = dict(final["spec_args"])
    if spec["model"] in TUNE_GRID:
        print(f"\n[4] {spec['model']} 제한 튜닝 {N_TUNE}개 설정")
        tuned = [R.run("4_tuning", **{**spec, "params": p}) for p in sample_grid(TUNE_GRID[spec["model"]], N_TUNE)]
        final = choose(final, tuned)
        spec = dict(final["spec_args"])
    else:
        print("\n[4] Ridge 는 RidgeCV 가 학습 fold 안에서 alpha 를 고른다. 별도 튜닝 생략")
    R.run("5_final_ref", m1_filled=True, **spec)
    print(f"  → 최종 설정: {final['spec']}")

    cv_table = R.table()
    cv_table.to_csv(C.OUTPUT_DIR / "cv_results.csv", index=False, encoding="utf-8-sig")

    # 5. 시험 구간 1회 평가 (선택 이후)
    print("\n[5] 시험 구간 평가")
    model_dev = train_model(dev, **spec)
    test_pred = predict_frame(model_dev, test)
    oof = R.oof[final["spec"]]
    sub = pd.concat([subgroup_table(oof, "dev_cv_oof"), subgroup_table(test_pred, "test")])
    sub.to_csv(C.OUTPUT_DIR / "subgroup_errors.csv", index=False, encoding="utf-8-sig")
    test_primary = metrics(test_pred[test_pred["process_complete"]])
    print("  시험 구간 공정 완전군:", {k: round(v, 3) for k, v in test_primary.items()})

    # 참고: 모델별 최선 설정의 시험 점수 (선택에 사용하지 않음)
    ref_rows = []
    for m, r in best_by_model.items():
        mdl = train_model(dev, **r["spec_args"])
        tp = predict_frame(mdl, test)
        ref_rows.append({"model": m, "spec": r["spec"], "cv_rmse_mean": r["cv_rmse_mean"],
                         **{f"test_{k}": v for k, v in metrics(tp[tp["process_complete"]]).items()}})
    ref = pd.DataFrame(ref_rows)
    ref.to_csv(C.OUTPUT_DIR / "model_comparison_test_reference.csv", index=False, encoding="utf-8-sig")

    pd.concat([oof.assign(set="dev_cv_oof"), test_pred.assign(set="test")]).to_csv(
        C.OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")

    # 목표 영역 사례 (Ni≥95 또는 Co≥90)
    both = pd.concat([oof.assign(set="dev_cv_oof"), test_pred.assign(set="test")])
    goal = both[(both["ni_recovery_pct"] >= C.TARGET_GOAL["ni_recovery_pct"])
                | (both["co_recovery_pct"] >= C.TARGET_GOAL["co_recovery_pct"])].copy()
    for t in C.TARGETS:
        goal[f"{t}_resid"] = goal[t] - goal[f"{t}_pred"]
    goal.to_csv(C.OUTPUT_DIR / "target_region_cases.csv", index=False, encoding="utf-8-sig")

    # 6. 해석 · 학습곡선 · 그림
    from .explain import learning_curve, make_figures, permutation_table
    print("\n[6] 중요도 · 학습곡선 · 그림")
    imp = permutation_table(model_dev, test[test["process_complete"]])
    imp.to_csv(C.OUTPUT_DIR / "permutation_importance_test.csv", index=False, encoding="utf-8-sig")
    lc = learning_curve(spec, dev)
    lc.to_csv(C.OUTPUT_DIR / "learning_curve.csv", index=False, encoding="utf-8-sig")
    make_figures(cv_table, oof, test_pred, lc, imp)

    # 7. 전체 420행 재학습 → Module 3 용 저장
    final_model = train_model(df, **spec)
    prim = oof[oof["process_complete"]]
    final_model.residual_rmse_ = [metrics(prim)["rmse_ni"], metrics(prim)["rmse_co"]]
    final_model.trained_rows_ = len(df)
    C.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, C.MODEL_PATH)

    summary = {
        "model_version": C.MODEL_VERSION, "data": info, "selected_policy_stage1": policy,
        "final_spec": final["spec"], "final_spec_args": spec,
        "cv_primary_rmse_mean": final["cv_rmse_mean"], "cv_primary_rmse_std": final["cv_rmse_std"],
        "cv_final_detail": {k: v for k, v in final.items() if k.startswith("primary") or k.startswith("all_rows")},
        "test_primary": test_primary, "test_all_rows": metrics(test_pred),
        "residual_rmse_for_predict_std": final_model.residual_rmse_,
        "model_path": str(C.MODEL_PATH.relative_to(C.ROOT)),
        "target_region_cases": int(len(goal)),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (C.OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print("\n저장:", C.OUTPUT_DIR.relative_to(C.ROOT), "/", C.MODEL_PATH.relative_to(C.ROOT))
    print(f"소요 {summary['runtime_sec']}초")


if __name__ == "__main__":
    main()
