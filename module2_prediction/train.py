"""Module 2 — 학습 · 튜닝 · 평가.

소유자: P3  (이 파일은 P3만 수정한다)

스펙 #5 요구사항:
  - Train / Validation / Test 분할 (analysis_date 기준 시간순)
  - RandomForest / XGBoost / LightGBM 비교 + 하이퍼파라미터 튜닝
  - RMSE / MAE / R^2 평가 + 과적합 확인
  - Feature Importance 분석
  - 실제 vs 예측 비교 그래프, 모델별 성능 비교표 (스펙 #6)

참고 기준선: 선형회귀만으로 R^2 = 0.887(Ni) / 0.902(Co), RMSE ≈ 1.9.
Arrhenius 항 추가 시 0.913. 트리 모델이 이보다 못하면 설정 오류다.
"""

import pandas as pd


def split(df: pd.DataFrame):
    """analysis_date 기준 시간순으로 train/valid/test 를 분할한다."""
    raise NotImplementedError


def train_model(train: pd.DataFrame):
    raise NotImplementedError


def evaluate(model, test: pd.DataFrame) -> dict:
    """{'rmse_ni':…, 'mae_ni':…, 'r2_ni':…, 'rmse_co':…, …} 를 반환한다."""
    raise NotImplementedError
