"""Module 1 — 특성 공학.

소유자: P2  (이 파일은 P2만 수정한다)
계약: docs/INTERFACE.md

파생변수 후보 (사전 검증된 상관계수, vs ni_recovery_pct):
  acid_per_feed   = acid_l / feed_kg            r = +0.506  (원본 acid_l 은 +0.461)
  impurity_total  = cu_pct + fe_pct + al_pct    r = -0.474
  temp_x_time     = temp_c * time_hr            R^2 기여 +0.013
  arrhenius       = exp(-Ea_R / (temp_c + 273.15))   Ea_R=3000 가정 시 +0.013
  ni_co_ratio     = ni_pct / co_pct             미검증 후보

주의: grade(원료등급) 더미변수는 성분 수치와 중복 정보다.
      추가해도 R^2 0.9003 → 0.9005 로 개선이 없으므로 인코딩에 시간을 쓰지 않는다.
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """정제된 프레임에 파생변수를 추가해 반환한다."""
    raise NotImplementedError
