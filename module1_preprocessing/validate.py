"""Module 1 — 데이터 품질 검증 및 시각화.

소유자: P1  (이 파일은 P1만 수정한다)

스펙 「2. 최종 결과물」의 "데이터 품질 검증 및 시각화 코드" 와
「3. 세부 시나리오」의 "전처리된 데이터의 정합성을 검증하고 시각화" 를 충족한다.

생성물 (docs/figures/)
---------------------
01_missing_heatmap.png        결측 위치 — 기간·등급 쏠림이 있는지 확인
02_before_after.png           정제 전후 분포 — 대치가 분포를 왜곡했는지 확인
03_range_violations.md        데이터사전 정상범위 위반 목록

정상범위는 원본 .xlsx 의 '데이터사전' 시트에서 직접 읽어온다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module1_preprocessing.cleaning import (
    COLUMN_MAP,
    FEATURE_COLS,
    SENTINELS,
    TARGET_COLS,
    clean,
    drop_duplicate_rows,
    load_raw,
    normalize_columns,
)

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

DICT_SHEET = "데이터사전"
FIG_DIR = Path("docs/figures")


def load_valid_ranges(path: str) -> dict[str, tuple[float, float]]:
    """'데이터사전' 시트의 '정상 범위/값' 열에서 수치 범위를 읽는다."""
    d = pd.read_excel(path, sheet_name=DICT_SHEET, dtype=str)
    col_kr, col_range = d.columns[0], d.columns[3]
    ranges: dict[str, tuple[float, float]] = {}
    for _, row in d.iterrows():
        eng = COLUMN_MAP.get(str(row[col_kr]).strip())
        text = str(row[col_range]).replace(" ", "")
        if eng is None or "~" not in text:
            continue
        lo, hi = text.split("~", 1)
        try:
            ranges[eng] = (float(lo), float(hi))
        except ValueError:
            continue
    return ranges


def plot_missing_heatmap(raw_renamed: pd.DataFrame, out: Path) -> dict:
    """정제 전 결측 위치를 격자 그림으로 그린다 (센티널도 결측으로 표시)."""
    cols = ["analysis_date", "grade"] + FEATURE_COLS + TARGET_COLS
    mask = raw_renamed[cols].isna()
    for col, vals in SENTINELS.items():
        mask[col] = mask[col] | raw_renamed[col].isin(vals)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [3, 1]}
    )
    ax.imshow(mask.to_numpy(), aspect="auto", cmap="gray_r", interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_ylabel("배치 (배치ID 순서 = 시간 순서)")
    ax.set_title(f"결측 위치  (검정 = 결측, 총 {int(mask.to_numpy().sum())}칸)")

    per_col = mask.sum().sort_values()
    ax2.barh(range(len(per_col)), per_col.to_numpy(), color="#4C78A8")
    ax2.set_yticks(range(len(per_col)))
    ax2.set_yticklabels(per_col.index, fontsize=8)
    ax2.set_title("컬럼별 결측 건수")
    for i, v in enumerate(per_col.to_numpy()):
        if v:
            ax2.text(v + 0.4, i, str(int(v)), va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)

    # 기간·등급 쏠림 진단
    n = len(raw_renamed)
    quarters = np.array_split(np.arange(n), 4)
    by_period = [int(mask.iloc[idx].to_numpy().sum()) for idx in quarters]
    by_grade = (
        mask.sum(axis=1)
        .groupby(raw_renamed["grade"].astype("string").str.strip().str.upper()
                 .str.replace("급", "", regex=False))
        .mean()
        .round(2)
        .to_dict()
    )
    return {"period": by_period, "grade": by_grade, "total": int(mask.to_numpy().sum())}


def plot_before_after(
    raw_renamed: pd.DataFrame, cleaned: pd.DataFrame, out: Path
) -> pd.DataFrame:
    """정제 전후 분포를 겹쳐 그려 대치로 인한 왜곡을 확인한다."""
    cols = FEATURE_COLS
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    rows = []
    for ax, col in zip(axes.ravel(), cols):
        before = pd.to_numeric(raw_renamed[col], errors="coerce")
        before = before[~before.isin(SENTINELS.get(col, []))].dropna()
        after = cleaned[col].dropna()
        lo = min(before.min(), after.min())
        hi = max(before.max(), after.max())
        bins = np.linspace(lo, hi, 26)
        ax.hist(before, bins=bins, alpha=0.55, label="정제 전(관측값)", color="#4C78A8")
        ax.hist(after, bins=bins, alpha=0.55, label="정제 후(대치 포함)", color="#E45756")
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=7)
        rows.append(
            {
                "컬럼": col,
                "관측 평균": round(before.mean(), 3),
                "정제후 평균": round(after.mean(), 3),
                "평균 변화율%": round(100 * (after.mean() - before.mean()) / before.mean(), 2),
                "관측 표준편차": round(before.std(), 3),
                "정제후 표준편차": round(after.std(), 3),
                "표준편차 변화율%": round(100 * (after.std() - before.std()) / before.std(), 2),
            }
        )
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("정제 전후 분포 비교 — 대치가 분포를 왜곡했는지 확인", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return pd.DataFrame(rows)


def range_violation_report(
    cleaned: pd.DataFrame, ranges: dict[str, tuple[float, float]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """데이터사전 정상범위를 벗어난 값을 집계한다."""
    summary, detail = [], []
    for col, (lo, hi) in ranges.items():
        if col not in cleaned.columns:
            continue
        bad = cleaned[(cleaned[col] < lo) | (cleaned[col] > hi)]
        summary.append(
            {"컬럼": col, "정상범위": f"{lo:g} ~ {hi:g}", "위반 건수": len(bad)}
        )
        for _, r in bad.iterrows():
            detail.append(
                {
                    "batch_id": r["batch_id"],
                    "컬럼": col,
                    "값": r[col],
                    "정상범위": f"{lo:g} ~ {hi:g}",
                    "대치값?": bool(r.get(f"imputed_{col}", False)),
                }
            )
    return pd.DataFrame(summary), pd.DataFrame(detail)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="정제 데이터 품질 검증")
    parser.add_argument(
        "--raw",
        default="data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx",
    )
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw(args.raw)
    raw_renamed = drop_duplicate_rows(normalize_columns(raw))
    for col in FEATURE_COLS + TARGET_COLS:
        raw_renamed[col] = pd.to_numeric(raw_renamed[col], errors="coerce")
    cleaned = clean(raw)
    ranges = load_valid_ranges(args.raw)

    print("=" * 66)
    print("1. 결측 히트맵")
    print("=" * 66)
    diag = plot_missing_heatmap(raw_renamed, FIG_DIR / "01_missing_heatmap.png")
    print(f"  결측 총계(센티널 포함) : {diag['total']}칸")
    print(f"  기간 4분할별 결측      : {diag['period']}  ← 한쪽에 몰리면 장비 이슈 의심")
    print(f"  등급별 행당 평균 결측  : {diag['grade']}")
    print(f"  저장                   : {FIG_DIR / '01_missing_heatmap.png'}")

    print()
    print("=" * 66)
    print("2. 정제 전후 분포 비교")
    print("=" * 66)
    stats = plot_before_after(raw_renamed, cleaned, FIG_DIR / "02_before_after.png")
    print(stats.to_string(index=False))
    worst = stats.loc[stats["평균 변화율%"].abs().idxmax()]
    print(f"\n  평균 변화 최대 : {worst['컬럼']} {worst['평균 변화율%']}%")
    worst_sd = stats.loc[stats["표준편차 변화율%"].abs().idxmax()]
    print(f"  표준편차 변화 최대 : {worst_sd['컬럼']} {worst_sd['표준편차 변화율%']}%")
    print(f"  저장 : {FIG_DIR / '02_before_after.png'}")

    print()
    print("=" * 66)
    print("3. 데이터사전 정상범위 위반")
    print("=" * 66)
    summary, detail = range_violation_report(cleaned, ranges)
    print(summary.to_string(index=False))
    print(f"\n  위반 합계 : {int(summary['위반 건수'].sum())}건")

    md = FIG_DIR / "03_range_violations.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# 데이터사전 정상범위 위반 리포트\n\n")
        f.write("`데이터사전` 시트의 '정상 범위/값' 열을 기준으로 집계했다.\n\n")
        f.write("## 컬럼별 요약\n\n")
        f.write(summary.to_markdown(index=False))
        f.write(f"\n\n**합계 {int(summary['위반 건수'].sum())}건**\n\n")
        f.write("## 상세\n\n")
        f.write(detail.to_markdown(index=False))
        f.write("\n")
    print(f"  저장 : {md}")


if __name__ == "__main__":
    main()
