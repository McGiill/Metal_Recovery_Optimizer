"""Module 3 설계 근거 — 원본 데이터 사실 정리 (MODULE3_INDEPENDENT_WORK 4.2 의 10개 질문).

소유자: P4  (이 파일은 P4만 수정한다)

    python -m module3_optimization.analysis.facts

결과는 요약 수치만 출력한다. 원본 행 자체는 공유하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from module3_optimization.analysis.load_raw import load_minimal
from module3_optimization.config import load_config
from module3_optimization.domain import build_domains
from module3_optimization.optimizers import make_grid

NI, CO = "ni_recovery_pct", "co_recovery_pct"
DECISION = ["acid_l", "temp_c", "time_hr"]


def section(title: str) -> None:
    print(f"\n## {title}")


def main() -> None:
    pd.set_option("display.width", 160)
    df = load_minimal()
    print(
        f"행 수 {len(df)} (중복 제거 후), 등급 {df['grade'].value_counts().to_dict()}"
    )

    section("1. 결정 변수 분포 (1 / 50 / 99 백분위)")
    q = df[DECISION + ["feed_kg"]].quantile([0.01, 0.5, 0.99]).T.round(2)
    print(q.to_string())

    section("2. 원료 1kg당 황산 (acid_per_feed)")
    r = df["acid_per_feed"].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).round(2)
    print(r.to_string())

    hit = (df[NI] >= 95) & (df[CO] >= 90)
    section("3. 목표 동시 달성 배치")
    cols = [
        "grade",
        "impurity_total",
        "feed_kg",
        "acid_per_feed",
        "temp_c",
        "time_hr",
        NI,
        CO,
    ]
    print(f"{hit.sum()}건 / 등급 {df.loc[hit, 'grade'].value_counts().to_dict()}")
    print(df.loc[hit, cols].round(2).to_string(index=False))
    print("전체 대비 백분위 순위 (%):")
    for c in ["acid_per_feed", "temp_c", "time_hr", "impurity_total"]:
        ranks = (df[c].rank(pct=True)[hit] * 100).round(0)
        print(f"  {c}: {ranks.tolist()}")

    section("4. 단독 달성 건수 — 병목")
    print(f"Ni ≥ 95: {(df[NI] >= 95).sum()}건, Co ≥ 90: {(df[CO] >= 90).sum()}건")

    section("5. 등급별 회수율 (평균 / 최대)")
    print(df.groupby("grade")[[NI, CO]].agg(["mean", "max"]).round(2).to_string())

    section("6. Ni–Co 회수율 상관")
    print(f"r = {df[[NI, CO]].corr().iloc[0, 1]:.3f}")

    section("7·9. 회수율과의 상관 (공정 변수 · 불순물)")
    corr_cols = ["acid_per_feed", "temp_c", "time_hr", "impurity_total"]
    print(df[corr_cols + [NI, CO]].corr().loc[corr_cols, [NI, CO]].round(2).to_string())

    section("8. 결정 변수 간 상관")
    print(df[["acid_per_feed", "temp_c", "time_hr"]].corr().round(2).to_string())

    section("10. 적용 범위 — 성공 배치 원료에서 Grid 점 중 범위 안 비율")
    cfg = load_config()
    grid = make_grid(cfg["decision_variables"], "coarse_step")
    feed_cols = ["ni_pct", "co_pct", "li_pct", "mn_pct", "cu_pct", "fe_pct", "al_pct"]
    for cols_name, cols in {
        "결정 변수 + 원료 (기본)": None,
        "결정 변수 + feed_kg": DECISION + ["feed_kg"],
    }.items():
        override = {"domain": {"columns": cols}}
        process, _ = build_domains(load_config(overrides=override), df)
        shares = []
        for _, row in df[hit].dropna(subset=feed_cols).iterrows():
            X = grid.assign(**{c: row[c] for c in feed_cols + ["feed_kg"]})
            in_dom = process.in_domain(X)
            corner = (X["temp_c"] >= 90) & (X["time_hr"] >= 7)
            shares.append((in_dom.mean(), in_dom[corner.to_numpy()].mean()))
        arr = np.array(shares)
        print(
            f"  {cols_name}: 전체 격자 {arr[:, 0].mean():.0%} · "
            f"고온장시간 구석(≥90℃, ≥7hr) {arr[:, 1].mean():.0%} 가 범위 안 "
            f"(학습 {process.n_train}행)"
        )


if __name__ == "__main__":
    main()
