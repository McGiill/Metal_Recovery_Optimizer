"""Module 1 정제 결과 → Module 2 학습 입력.

소유자: P3  (이 파일은 P3만 수정한다)

Module 1 의 batches.csv 는 전체 420행의 등급별 중앙값으로 이미 대치된 값이다.
그대로 학습·검증에 쓰면 검증 fold 의 정보가 대치값에 섞인다(누수).
imputed_<col> 플래그가 대치 직전의 결측 위치를 정확히 기록하므로, 그 칸을 NaN 으로
되돌린 뒤 대치는 학습 Pipeline 안에서 fold 별로 다시 한다.
"""

import numpy as np
import pandas as pd

from module1_preprocessing import cleaning as p1

from . import config as C


def load_module1(path=C.M1_BATCHES) -> pd.DataFrame:
    """batches.csv 를 읽어 대치 전 결측(NaN)을 복원한 프레임을 반환한다."""
    df = pd.read_csv(path, parse_dates=["analysis_date"])
    flags = [f"imputed_{c}" for c in C.FEATURE_COLS]
    missing = set(flags + C.FEATURE_COLS + C.TARGETS) - set(df.columns)
    if missing:
        raise ValueError(f"Module 1 출력에 필요한 열이 없음: {sorted(missing)}")
    if df[C.TARGETS].isna().any().any():
        raise ValueError("타깃 결측이 있음")

    df["grade"] = df["grade"].astype(str)
    df["batch_seq"] = df["batch_id"].str.extract(r"(\d+)$")[0].astype(int)
    for col in C.FEATURE_COLS:
        df[f"m1_filled_{col}"] = df[col]  # Module 1 전역 대치값 (비교 실험용)
        df.loc[df[f"imputed_{col}"].astype(bool), col] = np.nan

    df["process_complete"] = df[C.PROCESS].notna().all(axis=1)
    df["composition_complete"] = df[C.COMPOSITION].notna().all(axis=1)
    df["missing_temp_c"] = df["temp_c"].isna()
    df["missing_time_hr"] = df["time_hr"].isna()
    return df.sort_values("batch_seq").reset_index(drop=True)


def check_against_raw(df: pd.DataFrame, raw_path=C.RAW_XLSX) -> None:
    """복원한 NaN 위치·관측값이 원본에 Module 1 의 센티널 처리까지만 한 결과와 같은지 확인."""
    raw = p1.load_raw(str(raw_path))
    masked = p1.mask_sentinels(p1.drop_duplicate_rows(p1.normalize_columns(raw)))
    masked = masked.set_index("batch_id").loc[df["batch_id"]]
    for col in C.FEATURE_COLS:
        a = masked[col].astype(float).to_numpy()
        b = df[col].to_numpy()
        if not np.array_equal(np.isnan(a), np.isnan(b)) or not np.allclose(
            a[~np.isnan(a)], b[~np.isnan(b)]
        ):
            raise AssertionError(f"원본과 불일치: {col}")


def date_order_violations(df: pd.DataFrame) -> int:
    dated = df.dropna(subset=["analysis_date"]).sort_values("batch_seq")
    return int((dated["analysis_date"].diff().dt.days < 0).sum())


def split_dev_test(df: pd.DataFrame, test_start=C.TEST_START_DATE):
    """배치ID 순서 기준. test_start 이후 첫 배치부터 끝까지가 시험 구간."""
    if date_order_violations(df):
        raise ValueError("배치ID 순서와 날짜 순서가 다르므로 배치ID로 시간 분할할 수 없음")
    first = df.loc[df["analysis_date"] >= pd.Timestamp(test_start), "batch_seq"].min()
    is_test = df["batch_seq"] >= first
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
