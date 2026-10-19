"""해석 · 학습곡선 · 그림.

소유자: P3  (이 파일은 P3만 수정한다)

중요도는 원본 입력 단위의 permutation importance 이다. 열 하나를 섞으면 그 열에서 만든
파생 변수도 함께 바뀐다. 원본과 파생 변수가 상관되어 있어 중요도가 나뉘어 보일 수 있다.
상관·중요도는 예측 기여이지 공정의 인과 효과가 아니다.
SHAP 은 추가구현(#8) 후보로 남긴다.

대조 기준 (사전 측정된 상관 방향):
  temp_c +0.504 / acid_l +0.461 / ni_pct +0.420   (양)
  al_pct -0.463 / cu_pct -0.445 / fe_pct -0.411   (음)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from . import config as C
from .evaluation import metrics, predict_frame
from .model import RecoveryModel


def _rmse(model, frame):
    m = metrics(predict_frame(model, frame))
    return np.array([m["rmse_ni"], m["rmse_co"]])


def permutation_table(model: RecoveryModel, frame: pd.DataFrame, n_repeats=30, seed=C.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    base = _rmse(model, frame)
    rows = []
    for col in C.FEATURE_COLS:
        inc = []
        for _ in range(n_repeats):
            shuffled = frame.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            inc.append(_rmse(model, shuffled) - base)
        inc = np.array(inc)
        rows.append({"feature": col, "ni_rmse_increase": inc[:, 0].mean(), "ni_std": inc[:, 0].std(),
                     "co_rmse_increase": inc[:, 1].mean(), "co_std": inc[:, 1].std()})
    out = pd.DataFrame(rows)
    out["mean_increase"] = out[["ni_rmse_increase", "co_rmse_increase"]].mean(axis=1)
    out.attrs["n_rows"] = len(frame)
    return out.sort_values("mean_increase", ascending=False).reset_index(drop=True)


def learning_curve(spec: dict, dev: pd.DataFrame, fractions=(0.3, 0.5, 0.7, 0.85, 1.0), repeats=5):
    """마지막 CV fold 의 공정 완전 검증군을 고정하고 학습량만 바꾼다."""
    tr, va = list(TimeSeriesSplit(n_splits=C.N_CV_SPLITS).split(dev))[-1]
    train, val = dev.iloc[tr], dev.iloc[va]
    val = val[val["process_complete"]]
    rng = np.random.default_rng(C.RANDOM_STATE)
    rows = []
    for frac in fractions:
        for r in range(1 if frac == 1.0 else repeats):
            idx = np.sort(rng.choice(len(train), int(round(frac * len(train))), replace=False))
            sub = train.iloc[idx]
            model = RecoveryModel(**spec).fit(sub, sub[C.TARGETS])
            m = metrics(predict_frame(model, val))
            rows.append({"fraction": frac, "repeat": r, "n_train": model.n_train_,
                         "n_val": len(val), "rmse_mean": m["rmse_mean"]})
    return pd.DataFrame(rows)


def make_figures(cv_table, oof, test_pred, lc, imp, out_dir=C.FIGURE_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 정책 비교
    pol = cv_table[cv_table["stage"].str.startswith("1_policy")].sort_values("cv_rmse_mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(pol["spec"], pol["cv_rmse_mean"], xerr=pol["cv_rmse_std"], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("CV RMSE (mean of Ni/Co, process-complete val rows, %p)")
    ax.set_title("Missing-process policy comparison (LightGBM)")
    fig.tight_layout()
    fig.savefig(out_dir / "p3_01_policy_comparison.png", dpi=130)
    plt.close(fig)

    # 모델 × 특징 집합
    feat = cv_table[cv_table["stage"].isin(["1_policy", "2_features"]) & ~cv_table["m1_filled"]].copy()
    feat["model"] = feat["spec"].str.split("|").str[0]
    feat["fs"] = feat["spec"].str.split("|").str[2]
    piv = feat.pivot_table(index="model", columns="fs", values="cv_rmse_mean", aggfunc="min")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(piv.to_numpy(), cmap="viridis_r")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="CV RMSE (%p)")
    ax.set_title("Model x feature set (best policy per cell, CV)")
    fig.tight_layout()
    fig.savefig(out_dir / "p3_02_model_feature_grid.png", dpi=130)
    plt.close(fig)

    # 실제 vs 예측, 잔차
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for j, t in enumerate(C.TARGETS):
        for frame, label, color in ((oof, "dev CV (OOF)", "#999999"), (test_pred, "test", "#DD5555")):
            f = frame[frame["process_complete"].astype(bool)].dropna(subset=[f"{t}_pred"])
            axes[0, j].scatter(f[t], f[f"{t}_pred"], s=14, alpha=0.7, color=color, label=label)
            axes[1, j].scatter(f[f"{t}_pred"], f[t] - f[f"{t}_pred"], s=14, alpha=0.7, color=color, label=label)
        lo, hi = oof[t].min() - 2, oof[t].max() + 2
        axes[0, j].plot([lo, hi], [lo, hi], "k--", lw=1)
        axes[0, j].axvline(C.TARGET_GOAL[t], color="green", lw=0.8, ls=":")
        axes[0, j].set(xlabel=f"actual {t}", ylabel="predicted", title=f"{t}: actual vs predicted")
        axes[1, j].axhline(0, color="k", lw=1)
        axes[1, j].set(xlabel="predicted", ylabel="residual (actual - pred)", title=f"{t}: residuals")
        axes[0, j].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "p3_03_actual_vs_pred.png", dpi=130)
    plt.close(fig)

    # 학습곡선
    g = lc.groupby("fraction").agg(n=("n_train", "mean"), m=("rmse_mean", "mean"), s=("rmse_mean", "std"))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(g["n"], g["m"], yerr=g["s"].fillna(0), marker="o", capsize=3)
    ax.set(xlabel="training rows", ylabel="RMSE (mean Ni/Co, %p)",
           title=f"Learning curve (fixed last-fold val, n={int(lc['n_val'].iloc[0])})")
    fig.tight_layout()
    fig.savefig(out_dir / "p3_04_learning_curve.png", dpi=130)
    plt.close(fig)

    # 중요도
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(imp))
    ax.barh(y - 0.2, imp["ni_rmse_increase"], 0.4, xerr=imp["ni_std"], label="Ni")
    ax.barh(y + 0.2, imp["co_rmse_increase"], 0.4, xerr=imp["co_std"], label="Co")
    ax.set_yticks(y, imp["feature"])
    ax.invert_yaxis()
    ax.set(xlabel="RMSE increase when permuted (%p)",
           title=f"Permutation importance (test, process-complete, n={imp.attrs.get('n_rows', '?')})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "p3_05_permutation_importance.png", dpi=130)
    plt.close(fig)
