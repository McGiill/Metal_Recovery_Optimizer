"""Module 1 — 데이터 정제.

소유자: P1  (이 파일은 P1만 수정한다)
계약: docs/INTERFACE.md

원본 424행 × 16컬럼을 학습 가능한 형태로 정제한다.

처리 내역
---------
1. 컬럼명 영문 snake_case 변환 (COLUMN_MAP)
2. 완전 동일 행 제거 4건 → 420행
3. 분석일자 6종 포맷 → datetime (결측 22건은 NaT 유지)
4. 원료등급 12종 표기 → A / B / C
5. 센티널 9건 → NaN + sentinel_<col> 플래그
6. 결측 316칸 대치 + imputed_<col> 플래그
7. 표준화는 standardize() 로 분리 (학습 데이터로만 fit)

배치ID·분석일자는 데이터에 보존하되 FEATURE_COLS 에서 제외한다.
"""

from __future__ import annotations

import pandas as pd

SHEET_NAME = "원료분석_원본데이터"

COLUMN_MAP = {
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

# 모델 입력으로 쓰는 컬럼. batch_id(완전 식별자)와 analysis_date(시계열 분할용
# 메타데이터)는 데이터에 보존하되 여기서 제외한다. grade 는 대치에 쓰이지만
# 성분 수치와 중복 정보이므로(더미 추가 시 R^2 0.9003 → 0.9005) 피처에 넣지 않는다.
FEATURE_COLS = [
    "ni_pct",
    "co_pct",
    "li_pct",
    "mn_pct",
    "cu_pct",
    "fe_pct",
    "al_pct",
    "feed_kg",
    "acid_l",
    "temp_c",
    "time_hr",
]

TARGET_COLS = ["ni_recovery_pct", "co_recovery_pct"]

# 분석일자에 실제로 나타나는 6종 포맷. 순서대로 시도한다.
# NN.NN.NNNN 은 DD.MM.YYYY, NN/NN/NN 은 MM/DD/YY 로 확정했다.
# 근거: 배치ID가 날짜 오름차순이고 6종 포맷이 순환하므로
#       BP-0003(03.01.2025) · BP-0004(20250103) · BP-0005(Jan 03, 2025)
#       · BP-0006(01/03/25) 가 모두 2025-01-03 을 가리킨다.
DATE_FORMATS = [
    "%Y-%m-%d",   # 2025-01-02
    "%Y/%m/%d",   # 2025/01/02
    "%d.%m.%Y",   # 03.01.2025
    "%Y%m%d",     # 20250103
    "%b %d, %Y",  # Jan 03, 2025
    "%m/%d/%y",   # 01/03/25
]

# 컬럼별 센티널. 측정 실패를 나타내려고 넣은 특수값이며 실제 관측값이 아니다.
#   ni_pct  == 999     함량 단위가 wt% 이므로 999% 는 물리적으로 불가능
#   acid_l  == 15000   정상 범위 400~1500 의 정확히 10배 → 자릿수 입력 오류
#   temp_c  == -5      침출 반응이 -5℃ 에서 일어날 수 없음
#   time_hr == 0       반응 시간 0 인데 회수율 95.56% 기록(BP-2025-0231) → 모순
SENTINELS = {
    "ni_pct": [999.0],
    "acid_l": [15000.0],
    "temp_c": [-5.0],
    "time_hr": [0.0],
}

IMPUTE_STRATEGIES = ("grade_median", "median", "mean", "drop")


def load_raw(path: str) -> pd.DataFrame:
    """원본 .xlsx 의 '원료분석_원본데이터' 시트를 로드한다.

    분석일자는 포맷이 6종 섞여 있어 pandas 의 자동 추론이 틀릴 수 있으므로
    문자열로 읽어 parse_dates() 에서 명시적으로 변환한다.
    """
    return pd.read_excel(path, sheet_name=SHEET_NAME, dtype={"분석일자": str})


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명을 영문 snake_case 로 바꾼다."""
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"원본에 없는 컬럼: {sorted(missing)}")
    return df.rename(columns=COLUMN_MAP)


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """완전히 동일한 행을 제거한다 (원본 4건, 424 → 420행)."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def parse_dates(series: pd.Series) -> pd.Series:
    """6종 포맷을 순서대로 시도해 datetime 으로 변환한다.

    어느 포맷에도 맞지 않으면 NaT 로 남긴다 (원본 결측 22건 포함).
    """
    raw = series.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        todo = out.isna() & raw.notna() & (raw != "")
        if not todo.any():
            break
        parsed = pd.to_datetime(raw[todo], format=fmt, errors="coerce")
        out.loc[todo] = parsed
    return out


def normalize_grade(series: pd.Series) -> pd.Series:
    """원료등급 12종 표기를 A / B / C 로 통일한다.

    공백 패딩(' A '), 소문자('a'), 한글 접미사('A급') 를 모두 흡수한다.
    """
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace("급", "", regex=False)
        .str.strip()
    )
    unknown = set(cleaned.dropna().unique()) - {"A", "B", "C"}
    if unknown:
        raise ValueError(f"정규화 후에도 알 수 없는 등급: {sorted(unknown)}")
    return cleaned.astype("category")


def mask_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """센티널을 NaN 으로 바꾸고 sentinel_<col> 플래그를 추가한다."""
    out = df.copy()
    for col, values in SENTINELS.items():
        hit = out[col].isin(values)
        out[f"sentinel_{col}"] = hit
        out.loc[hit, col] = pd.NA
    return out


def impute(df: pd.DataFrame, strategy: str = "grade_median") -> pd.DataFrame:
    """결측을 채우고 imputed_<col> 플래그를 추가한다.

    strategy
    --------
    grade_median : 같은 원료등급(A/B/C) 내 중앙값. 기본값.
        등급이 성분 분포를 규정하므로(Ni 함량 평균 A 24.31 / B 18.14 / C 12.25)
        전체 중앙값으로 채우면 A급 결측이 하향 왜곡된다.
    median / mean : 전체 중앙값 / 평균
    drop         : 결측이 있는 행을 제거 (420 → 200행)
    """
    if strategy not in IMPUTE_STRATEGIES:
        raise ValueError(f"strategy 는 {IMPUTE_STRATEGIES} 중 하나여야 한다")

    out = df.copy()
    numeric = FEATURE_COLS + TARGET_COLS
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[f"imputed_{col}"] = out[col].isna()

    if strategy == "drop":
        keep = out[numeric].notna().all(axis=1)
        return out.loc[keep].reset_index(drop=True)

    for col in numeric:
        if not out[col].isna().any():
            continue
        if strategy == "grade_median":
            filler = out.groupby("grade", observed=True)[col].transform("median")
            out[col] = out[col].fillna(filler)
            # 어떤 등급 전체가 결측이면 전체 중앙값으로 보정 (현 데이터에선 미발생)
            out[col] = out[col].fillna(out[col].median())
        elif strategy == "median":
            out[col] = out[col].fillna(out[col].median())
        else:
            out[col] = out[col].fillna(out[col].mean())
    return out


def clean(df: pd.DataFrame, impute_strategy: str = "grade_median") -> pd.DataFrame:
    """원본 프레임을 계약 스키마로 정제한다."""
    out = normalize_columns(df)
    out = drop_duplicate_rows(out)
    out["analysis_date"] = parse_dates(out["analysis_date"])
    out["grade"] = normalize_grade(out["grade"])
    out = mask_sentinels(out)
    out = impute(out, strategy=impute_strategy)
    ordered = (
        ["batch_id", "analysis_date", "grade"]
        + FEATURE_COLS
        + TARGET_COLS
        + [c for c in out.columns if c.startswith(("sentinel_", "imputed_"))]
    )
    return out[ordered]


def standardize(train: pd.DataFrame, *others: pd.DataFrame):
    """FEATURE_COLS 를 표준화한다 (평균 0, 표준편차 1).

    스케일러는 **학습 데이터로만 fit** 한다. 검증·테스트까지 포함해 fit 하면
    그쪽 분포 정보가 학습에 새어 들어간다(data leakage).

    트리 기반 모델(RandomForest·XGBoost·LightGBM)은 분기 기준이 크기에
    무관하므로 표준화가 필요 없다. 선형 기준선 비교와 스펙 요구사항
    (「2. 최종 결과물」의 "데이터 정규화/표준화 코드") 때문에 구현한다.

    Returns
    -------
    (scaler, 변환된 train, 변환된 others...)
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train[FEATURE_COLS])

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[FEATURE_COLS] = scaler.transform(frame[FEATURE_COLS])
        return out

    return (scaler, apply(train), *(apply(f) for f in others))


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="원료 분석 데이터 정제")
    parser.add_argument(
        "--raw",
        default="data/raw/영풍그룹_금속회수율최적화_RawMaterialData_학생배포용.xlsx",
    )
    parser.add_argument("--out", default="data/processed/batches.csv")
    parser.add_argument(
        "--impute", default="grade_median", choices=IMPUTE_STRATEGIES
    )
    args = parser.parse_args()

    raw = load_raw(args.raw)
    print(f"원본            : {len(raw)}행 × {raw.shape[1]}컬럼")

    cleaned = clean(raw, impute_strategy=args.impute)
    print(f"정제 후         : {len(cleaned)}행 × {cleaned.shape[1]}컬럼")
    print(f"대치 전략       : {args.impute}")

    sentinel_cols = [c for c in cleaned.columns if c.startswith("sentinel_")]
    imputed_cols = [c for c in cleaned.columns if c.startswith("imputed_")]
    print(f"센티널 처리     : {int(cleaned[sentinel_cols].sum().sum())}칸")
    print(f"대치된 칸       : {int(cleaned[imputed_cols].sum().sum())}칸")
    print(f"날짜 파싱 실패  : {int(cleaned['analysis_date'].isna().sum())}건 (원본 결측)")
    print(f"등급 분포       : {cleaned['grade'].value_counts().to_dict()}")
    remaining = int(cleaned[FEATURE_COLS + TARGET_COLS].isna().sum().sum())
    print(f"남은 결측       : {remaining}칸")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    print(f"저장            : {out_path}")


if __name__ == "__main__":
    main()
