"""Module 2 설정.

소유자: P3  (이 파일은 P3만 수정한다)
"""

from pathlib import Path

from module1_preprocessing.cleaning import FEATURE_COLS, TARGET_COLS

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx"
M1_BATCHES = ROOT / "data/processed/batches.csv"  # python -m module1_preprocessing.cleaning
OUTPUT_DIR = ROOT / "data/processed/p3_prediction"
FIGURE_DIR = ROOT / "docs/figures"
MODEL_PATH = ROOT / "artifacts/recovery_model.joblib"

MODEL_VERSION = "p3-v0.1"
RANDOM_STATE = 42

COMPOSITION = ["ni_pct", "co_pct", "li_pct", "mn_pct", "cu_pct", "fe_pct", "al_pct"]
PROCESS = ["feed_kg", "acid_l", "temp_c", "time_hr"]
DECISION = ["acid_l", "temp_c", "time_hr"]  # Module 3 결정 변수
assert COMPOSITION + PROCESS == FEATURE_COLS
TARGETS = list(TARGET_COLS)  # [ni_recovery_pct, co_recovery_pct]

# 원본 11개 + Module 1(P2) 파생 변수. 정의는 module1_preprocessing.features.add_features.
FEATURE_SETS = {
    "F0": [],
    "F1": ["acid_time_interaction"],
    "F2": ["acid_time_interaction", "acid_per_feed"],
    "F3": ["acid_time_interaction", "acid_temp_interaction"],
    "F4": ["arrhenius"],  # P2 판정: 선형 모델에서만 개선
}

# P2 노트북이 선택에 쓰지 않은 후기 구간 시작일. 이 날짜 이후 배치를 시험 구간으로 둔다.
# 배치ID 순서 = 날짜 순서(역전 0건)이므로 날짜 결측 배치도 배치ID로 위치를 정한다.
TEST_START_DATE = "2025-09-16"
N_CV_SPLITS = 4

TARGET_GOAL = {"ni_recovery_pct": 95.0, "co_recovery_pct": 90.0}
