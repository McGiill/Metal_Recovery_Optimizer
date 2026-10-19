"""시간순 전방 CV, 지표, 하위군 집계.

소유자: P3  (이 파일은 P3만 수정한다)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from . import config as C
from .model import RecoveryModel

TRACK_COLS = ["batch_id", "batch_seq", "grade", "process_complete", "composition_complete",
              "missing_temp_c", "missing_time_hr"]


def use_m1_filled(df: pd.DataFrame) -> pd.DataFrame:
    """비교용: Module 1 이 전체 420행으로 대치한 값을 그대로 쓴 프레임 (누수 포함 참고 기준)."""
    out = df.copy()
    for col in C.FEATURE_COLS:
        out[col] = out[f"m1_filled_{col}"]
    return out


def predict_frame(model: RecoveryModel, frame: pd.DataFrame, fold=-1) -> pd.DataFrame:
    pred = model.predict(frame)
    out = frame[TRACK_COLS + C.TARGETS].copy()
    for j, t in enumerate(C.TARGETS):
        out[f"{t}_pred"] = pred[:, j]
    out["fold"] = fold
    out["n_train"] = model.n_train_
    return out


def forward_cv(make_model, dev: pd.DataFrame, n_splits=C.N_CV_SPLITS, m1_filled=False):
    """확장 윈도 전방 CV. 대치·파생·스케일·모델은 매 fold 의 train 에서만 fit 한다.

    m1_filled=True 는 Module 1 전역 대치값을 그대로 쓰는 참고 실험이다.
    """
    data = use_m1_filled(dev) if m1_filled else dev
    parts = []
    for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(data)):
        train, val = data.iloc[tr], data.iloc[va]
        model = make_model().fit(train, train[C.TARGETS])
        frame = predict_frame(model, val, fold)
        # 하위군 표시는 대치 전 결측 기준으로 유지
        frame[["process_complete", "missing_temp_c", "missing_time_hr", "composition_complete"]] = \
            dev.iloc[va][["process_complete", "missing_temp_c", "missing_time_hr",
                          "composition_complete"]].to_numpy()
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def metrics(frame: pd.DataFrame) -> dict:
    n = len(frame)
    covered = frame.dropna(subset=[f"{t}_pred" for t in C.TARGETS])
    res = {"n_eval": n, "coverage": len(covered) / n if n else np.nan}
    if len(covered) < 2:
        return res
    rmses = []
    for t in C.TARGETS:
        key = t.split("_")[0]  # ni / co
        y, p = covered[t], covered[f"{t}_pred"]
        res[f"rmse_{key}"] = float(np.sqrt(mean_squared_error(y, p)))
        res[f"mae_{key}"] = float(mean_absolute_error(y, p))
        res[f"r2_{key}"] = float(r2_score(y, p))
        rmses.append(res[f"rmse_{key}"])
    res["rmse_mean"] = float(np.mean(rmses))
    return res


SUBGROUPS = {
    "primary_process_complete": lambda f: f["process_complete"].astype(bool),
    "all_rows": lambda f: pd.Series(True, index=f.index),
    "temp_missing_only": lambda f: f["missing_temp_c"].astype(bool) & ~f["missing_time_hr"].astype(bool),
    "time_missing_only": lambda f: f["missing_time_hr"].astype(bool) & ~f["missing_temp_c"].astype(bool),
    "temp_and_time_missing": lambda f: f["missing_temp_c"].astype(bool) & f["missing_time_hr"].astype(bool),
    "composition_imputed": lambda f: ~f["composition_complete"].astype(bool),
}


def summarize(oof: pd.DataFrame, name: str) -> dict:
    """주 지표: fold 별 공정 완전군 Ni·Co 평균 RMSE 의 평균·표준편차."""
    per_fold = [metrics(f[f["process_complete"].astype(bool)]).get("rmse_mean", np.nan)
                for _, f in oof.groupby("fold")]
    out = {"spec": name, "cv_rmse_mean": float(np.mean(per_fold)),
           "cv_rmse_std": float(np.std(per_fold)), "fold_rmse": per_fold,
           "n_train_last_fold": int(oof["n_train"].iloc[-1])}
    for sub, sel in SUBGROUPS.items():
        for k, v in metrics(oof[sel(oof)]).items():
            out[f"{sub}.{k}"] = v
    return out


def paired_better(cand: dict, base: dict, min_wins: int = 3) -> bool:
    """복잡한 후보는 평균 RMSE 가 낮고 4 fold 중 min_wins 이상에서 이길 때만 채택한다."""
    c, b = np.array(cand["fold_rmse"]), np.array(base["fold_rmse"])
    return bool(c.mean() < b.mean() and (c < b).sum() >= min_wins)


def subgroup_table(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = [{"set": label, "subgroup": s, **metrics(frame[sel(frame)])} for s, sel in SUBGROUPS.items()]
    rows += [{"set": label, "subgroup": f"grade_{g}", **metrics(f)} for g, f in frame.groupby("grade")]
    return pd.DataFrame(rows)
