"""배치 추론 · 혼합 원료 예시.

소유자: P3  (이 파일은 P3만 수정한다)

Module 3 는 joblib 으로 RecoveryModel 을 불러 predict() 를 직접 쓸 수 있다 (RecoveryModelAdapter).
predict_batch() 는 같은 모델에 입력 검증·적용 범위 상태를 덧붙인 화면·API 용 함수다.

status
  ok             정상 예측
  out_of_domain  학습 자료 1~99 백분위 범위 밖 입력이 있음 → 예측은 하되 정상 추천으로 쓰지 않는다
  invalid_input  필수 값 누락 또는 물리적으로 불가능한 값 → 예측하지 않음
"""

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data import load_module1
from .model import RecoveryModel


def load_model(path=C.MODEL_PATH) -> RecoveryModel:
    return joblib.load(path)


def predict_batch(model: RecoveryModel, candidates: pd.DataFrame, mode: str = "optimization") -> pd.DataFrame:
    """행 순서·index 를 보존해 Ni·Co 예측과 상태를 반환한다.

    mode="optimization": 공정 4변수가 모두 필요 (추천 후보 평가)
    mode="historical"  : 공정 값이 빠진 과거 배치도 모델 정책이 허용하면 예측
    """
    if mode not in ("optimization", "historical"):
        raise ValueError(mode)
    X = candidates.copy()
    for col in C.FEATURE_COLS:
        X[col] = pd.to_numeric(X.get(col), errors="coerce")
    status = pd.Series("ok", index=X.index, dtype=object)
    warn = pd.Series([[] for _ in range(len(X))], index=X.index, dtype=object)

    def flag(mask, state, msg):
        for i in X.index[mask]:
            warn[i].append(msg)
            if state == "invalid_input" or status[i] == "ok":
                status[i] = state

    need = C.PROCESS if mode == "optimization" else ["feed_kg"]
    flag(X[need].isna().any(axis=1).to_numpy(), "invalid_input", "필수 공정 값 누락")
    flag((X[C.COMPOSITION] < 0).any(axis=1) | (X[C.COMPOSITION].sum(axis=1) > 100), "invalid_input",
         "성분 값이 0~100 wt% 범위를 벗어남")
    flag(((X[C.PROCESS] <= 0) & X[C.PROCESS].notna()).any(axis=1), "invalid_input", "공정 값이 0 이하")
    no_grade = ("grade" not in X) or X["grade"].isna()
    flag(np.asarray(no_grade & X[C.COMPOSITION].isna().any(axis=1)), "invalid_input",
         "성분 결측인데 등급 정보 없음")

    rng = model.feature_range_
    for col in C.FEATURE_COLS:
        v = X[col]
        out = (v < rng.loc[col, "p01"]) | (v > rng.loc[col, "p99"])
        flag(out.to_numpy(), "out_of_domain", f"{col} 학습 범위(1~99%) 밖")

    pred = np.full((len(X), 2), np.nan)
    ok = (status != "invalid_input").to_numpy()
    if ok.any():
        pred[ok] = model.predict(X.loc[ok])
    rejected = ok & np.isnan(pred).any(axis=1)
    flag(rejected, "invalid_input", "모델 정책상 공정 결측 입력은 예측하지 않음")

    return pd.DataFrame({
        "ni_recovery_pct_pred": pred[:, 0], "co_recovery_pct_pred": pred[:, 1],
        "status": status, "warnings": warn.map("; ".join), "model_version": model.model_version,
    }, index=X.index)


def mix_materials(a: dict, b: dict, ratio_a: float, total_feed_kg: float,
                  composition_fill: dict | None = None) -> dict:
    """같은 질량 기준 성분 가중 평균. 혼합 등급은 임의로 만들지 않는다(grade=None).

    결측 성분은 composition_fill[원료등급][성분] (학습 자료 등급별 중앙값) 으로 원료별로 먼저 채운다.
    혼합물의 회수율 실측이 없으므로 이 결과는 추론 예시이지 정확도 검증이 아니다.
    """
    def complete(m):
        m = dict(m)
        for col in C.COMPOSITION:
            if m.get(col) is None or pd.isna(m.get(col)):
                if not composition_fill or m.get("grade") not in composition_fill:
                    raise ValueError(f"{col} 결측을 채울 등급 정보가 없음")
                m[col] = composition_fill[m["grade"]][col]
        return m

    a, b = complete(a), complete(b)
    mix = {col: ratio_a * a[col] + (1 - ratio_a) * b[col] for col in C.COMPOSITION}
    mix.update(feed_kg=total_feed_kg, grade=None)
    return mix


def grade_medians(model: RecoveryModel) -> dict:
    """모델 학습 시 사용한 등급별 성분 중앙값 (혼합 예시용)."""
    imp = model.pipelines_[C.TARGETS[0]].named_steps["impute"]
    return imp.grade_median_[C.COMPOSITION].to_dict(orient="index")


def demo() -> None:
    model = load_model()
    df = load_module1()  # 대치 전 NaN 복원본
    base = df[df["process_complete"] & df["composition_complete"]].head(3)
    cand = base[["grade"] + C.FEATURE_COLS].copy()
    cand.loc[cand.index[1], "temp_c"] = np.nan          # 공정 값 누락 → invalid_input
    cand.loc[cand.index[2], "acid_l"] = 3000            # 학습 범위 밖 → out_of_domain
    print(pd.concat([cand[C.DECISION], predict_batch(model, cand)], axis=1).to_string())

    a = df[df["grade"] == "A"].iloc[0].to_dict()
    b = df[df["grade"] == "C"].iloc[0].to_dict()
    mix = mix_materials(a, b, 0.7, total_feed_kg=150.0, composition_fill=grade_medians(model))
    rows = pd.DataFrame([{**mix, "acid_l": acid, "temp_c": 85.0, "time_hr": 6.0} for acid in (800, 1100, 1400)])
    print("\nA:C = 7:3 혼합, 원료 150kg, 85℃, 6h")
    print(pd.concat([rows[["acid_l"]], predict_batch(model, rows)], axis=1).to_string())


if __name__ == "__main__":
    demo()
