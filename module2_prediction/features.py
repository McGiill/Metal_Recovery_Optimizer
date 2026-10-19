"""fold 내부에서 fit 되는 대치·특징 변환기.

소유자: P3  (이 파일은 P3만 수정한다)
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from module1_preprocessing.features import add_features

from . import config as C


class GradeMedianImputer(BaseEstimator, TransformerMixin):
    """등급별 중앙값 대치 (Module 1 과 같은 방법). 학습 데이터에서만 중앙값을 계산한다.

    등급·컬럼 관측값이 없거나, 처음 보는 등급이거나, grade 열이 없으면 학습 전역 중앙값.
    전역 중앙값도 없으면 0 으로 채우지 않고 오류를 낸다.
    """

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        cols = list(self.columns)
        self.global_median_ = X[cols].median()
        empty = self.global_median_[self.global_median_.isna()].index.tolist()
        if empty:
            raise ValueError(f"학습 데이터에서 전부 결측인 컬럼: {empty}")
        self.grade_median_ = X.groupby("grade")[cols].median() if "grade" in X else None
        return self

    def transform(self, X):
        cols = list(self.columns)
        X = X.copy()
        if not cols or not X[cols].isna().any().any():
            return X
        if self.grade_median_ is not None and "grade" in X:
            by_grade = self.grade_median_.reindex(X["grade"].astype(str).to_numpy())
            by_grade.index = X.index
            unknown = ~X["grade"].astype(str).isin(self.grade_median_.index)
            if (unknown & X[cols].isna().any(axis=1)).any():
                warnings.warn("등급 정보 없이 결측 대치: 학습 전역 중앙값 사용", stacklevel=2)
            fill = by_grade.fillna(self.global_median_)
        else:
            warnings.warn("grade 열이 없어 학습 전역 중앙값으로 대치", stacklevel=2)
            fill = pd.DataFrame([self.global_median_] * len(X), index=X.index)
        X[cols] = X[cols].fillna(fill[cols])
        return X


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """원본 11개 + 선택한 파생 변수 행렬. 등급·플래그·ID 는 입력에서 제외한다.

    파생 변수는 Module 1 add_features() 를 그대로 쓴다. 공정 값이 NaN 이면 파생 값도 NaN.
    """

    def __init__(self, derived=()):
        self.derived = derived

    def fit(self, X, y=None):
        self.feature_names_ = C.FEATURE_COLS + list(self.derived)
        return self

    def transform(self, X):
        out = add_features(X[C.FEATURE_COLS], feature_cols=list(self.derived)) if self.derived else X
        return out[self.feature_names_].astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
