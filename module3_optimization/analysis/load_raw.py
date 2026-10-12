"""원본 최소 정리 — Module 3 설계 근거 분석 전용.

소유자: P4  (이 파일은 P4만 수정한다)

⚠ 정제 산출물이 아니다. P1 의 정제 파이프라인(module1_preprocessing/cleaning.py)을 대체하지 않는다.
  - 중복 제거, 등급 표기 통일, 센티널 → NaN 만 한다
  - 결측 대치는 하지 않는다 (분석은 필요한 컬럼만 dropna)
  - 날짜는 Module 3 에서 쓰지 않으므로 파싱하지 않는다
P1 정제 데이터(data/processed/)가 생기면 그쪽으로 교체한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = (
    REPO_ROOT
    / "data"
    / "raw"
    / "영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx"
)
SHEET = "원료분석_원본데이터"

# 저장소 뼈대(module1_preprocessing)의 snake_case 컬럼명에 맞춘다.
RENAME = {
    "배치ID": "batch_id",
    "분석일자": "analysis_date",
    "원료등급": "grade",
    "Ni_함량_pct": "ni_pct",
    "Co_함량_pct": "co_pct",
    "Li_함량_pct": "li_pct",
    "Mn_함량_pct": "mn_pct",
    "Cu_불순물_pct": "cu_pct",
    "Fe_불순물_pct": "fe_pct",
    "Al_불순물_pct": "al_pct",
    "원료투입량_kg": "feed_kg",
    "황산투입량_L": "acid_l",
    "침출온도_C": "temp_c",
    "침출시간_hr": "time_hr",
    "Ni_회수율_pct": "ni_recovery_pct",
    "Co_회수율_pct": "co_recovery_pct",
}

# README 기준 센티널 9건
SENTINELS = {"ni_pct": 999, "acid_l": 15000, "temp_c": -5, "time_hr": 0}


def load_minimal(path: str | Path = RAW_PATH) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=SHEET)
    missing = set(RENAME) - set(raw.columns)
    if missing:
        raise ValueError(f"원본 컬럼 누락: {sorted(missing)}")

    df = raw.drop_duplicates().rename(columns=RENAME).reset_index(drop=True)
    df["grade"] = (
        df["grade"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("급", "", regex=False)
    )
    for col, value in SENTINELS.items():
        df[col] = df[col].astype(float).mask(df[col] == value, np.nan)

    df["acid_per_feed"] = df["acid_l"] / df["feed_kg"]
    df["impurity_total"] = df["cu_pct"] + df["fe_pct"] + df["al_pct"]
    return df
