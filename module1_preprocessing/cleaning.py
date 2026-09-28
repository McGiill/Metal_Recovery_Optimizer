"""Module 1 — 데이터 정제.

소유자: P1  (이 파일은 P1만 수정한다)
계약: docs/INTERFACE.md

처리 대상 (원본 420행 기준, 중복 4건 제거 후):
  - 분석일자 10종 포맷 → ISO YYYY-MM-DD (결측 22건 허용)
  - 원료등급 12종 표기 → A / B / C
  - 센티널 9건 → NaN 후 대치, sentinel_<col> 플래그 유지
      ni_pct == 999 (3건) / acid_l == 15000 (2건)
      temp_c == -5  (2건) / time_hr == 0    (2건)
  - 결측 307셀 → 대치 + imputed_<col> 플래그
  - 배치ID 중복 4건 → 전부 완전 동일 행이므로 drop_duplicates()
"""

import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    """원본 .xlsx 의 '원료분석_원본데이터' 시트를 로드한다."""
    raise NotImplementedError


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """날짜·등급·센티널·결측·중복을 처리해 계약 스키마로 반환한다."""
    raise NotImplementedError
