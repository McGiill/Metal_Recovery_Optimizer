"""P1 정제본의 결측 보존과 P2 시간순 평가 연결 회귀 테스트."""

import numpy as np
import pandas as pd
import pytest

from module1_preprocessing import cleaning as p1
from module1_preprocessing.features import (
    ALL_DERIVED_COLS,
    _feature_pair,
    add_features,
    evaluate_features,
    load_p1_output,
    prepare_processed,
)


@pytest.fixture
def cleaned():
    count = 60
    step = np.arange(count)
    frame = pd.DataFrame({
        'batch_id': [f'B{i:03d}' for i in step],
        'analysis_date': pd.date_range('2025-01-01', periods=count).strftime('%Y-%m-%d'),
        'grade': np.resize(['A', 'B', 'C'], count),
        'ni_pct': 15. + step / 10,
        'co_pct': 5. + step / 30,
        'li_pct': 3., 'mn_pct': 8., 'cu_pct': .5, 'fe_pct': .8, 'al_pct': 1.,
        'feed_kg': 100. + step / 5,
        'acid_l': 800. + step * 5,
        'temp_c': 50. + step / 2,
        'time_hr': 3. + step / 30,
        'ni_recovery_pct': 80. + step / 5,
        'co_recovery_pct': 75. + step / 5,
    })
    frame.loc[[1, 4, 7, 10], 'ni_pct'] = np.nan
    frame.loc[2, 'acid_l'] = np.nan
    frame.loc[5, 'temp_c'] = -5.  # 센티널도 공정 결측으로 전달되어야 한다.
    frame.loc[8, 'time_hr'] = np.nan
    frame.loc[11, 'feed_kg'] = np.nan
    frame.loc[13, 'analysis_date'] = None
    frame.loc[14, 'ni_recovery_pct'] = np.nan
    return p1.clean(frame.rename(columns={v: k for k, v in p1.COLUMN_MAP.items()}))


def test_fresh_p1_csv_runs_feature_evaluation(cleaned, tmp_path):
    original = cleaned.copy(deep=True)
    path = tmp_path / 'p1_processed.csv'
    cleaned.to_csv(path, index=False, encoding='utf-8-sig')
    loaded = load_p1_output(path)
    assert loaded[p1.PROCESS_COLS].isna().sum().sum() == 4
    featured = add_features(loaded, ALL_DERIVED_COLS)
    assert pd.isna(featured.loc[5, 'arrhenius'])
    assert pd.isna(featured.loc[2, 'acid_per_feed'])

    eligible, review = prepare_processed(loaded)
    assert len(eligible) == 54
    assert set(review['batch_id']) == {'B002', 'B005', 'B008', 'B011', 'B013', 'B014'}
    assert eligible['ni_pct'].isna().sum() == 4
    assert eligible[p1.PROCESS_COLS + p1.TARGET_COLS].notna().all().all()
    again, no_more_review = prepare_processed(eligible)
    pd.testing.assert_frame_equal(again, eligible)
    assert no_more_review.empty

    # CSV 로딩뿐 아니라 DataFrame 직접 평가도 같은 결측 규칙을 사용한다.
    result = evaluate_features(loaded, n_splits=2, model_names=['Ridge'],
                               derived_cols=ALL_DERIVED_COLS)
    assert len(result['review']) == 6
    assert len(result['cv_detail']) == 26 * 2
    assert np.isfinite(result['cv_detail'].select_dtypes('number')).all().all()
    assert np.isfinite(result['test_scores'].select_dtypes('number')).all().all()
    assert (result['folds']['train_end'] < result['folds']['valid_start']).all()
    assert result['development']['analysis_date'].max() < result['test']['analysis_date'].min()
    pd.testing.assert_frame_equal(cleaned, original)
    assert loaded['ni_pct'].notna().all()  # EDA용 P1 대치값은 그대로 보존한다.


def test_old_filled_process_and_target_flags_are_restored(cleaned):
    frame = cleaned.copy()
    frame[p1.PROCESS_COLS] = frame[p1.PROCESS_COLS].fillna(999.)
    # 문자열 False를 bool('False')로 해석하면 모든 행이 사라지는 것을 방지한다.
    for col in p1.FEATURE_COLS + p1.TARGET_COLS:
        frame[f'imputed_{col}'] = frame[f'imputed_{col}'].astype(str)
    eligible, review = prepare_processed(frame)
    assert len(eligible) == 54
    assert pd.isna(review.loc[2, 'acid_l'])
    assert pd.isna(review.loc[14, 'ni_recovery_pct'])


def test_imputation_uses_only_training_observations(cleaned):
    train = cleaned.iloc[:4].copy()
    train['grade'] = ['A', 'A', 'B', 'B']
    train['ni_pct'] = [2., 4., 10., np.nan]
    train[p1.PROCESS_COLS] = [100., 900., 70., 4.]
    other = train.iloc[:3].copy()
    other.index = [20, 21, 22]
    other['grade'] = ['A', 'C', 'A']
    other['ni_pct'] = [np.nan, np.nan, 10000.]
    before = train.copy(deep=True)
    xtrain, xother = _feature_pair(train, other, ALL_DERIVED_COLS)
    assert xtrain['ni_pct'].tolist() == [2., 4., 10., 10.]
    # A급 중앙값=3, 학습에서 보지 못한 C급은 관측값 전체 중앙값=4.
    assert xother['ni_pct'].tolist() == [3., 4., 10000.]
    assert xother.loc[20, 'ni_co_ratio'] == pytest.approx(3. / other.loc[20, 'co_pct'])
    pd.testing.assert_frame_equal(train, before)


def test_empty_training_composition_is_reported(cleaned):
    eligible, _ = prepare_processed(cleaned)
    train = eligible.iloc[:10].copy()
    train['ni_pct'] = np.nan
    with pytest.raises(ValueError, match='학습 구간에서 전부 결측인 성분.*ni_pct'):
        _feature_pair(train, eligible.iloc[-5:], ALL_DERIVED_COLS)


@pytest.mark.parametrize('value', [np.inf, -np.inf])
def test_p1_loader_rejects_infinity(cleaned, tmp_path, value):
    cleaned.loc[0, 'temp_c'] = value
    path = tmp_path / 'p1_processed.csv'
    cleaned.to_csv(path, index=False)
    with pytest.raises(ValueError, match='무한값'):
        load_p1_output(path)


@pytest.mark.parametrize('value', ['unknown', None])
def test_invalid_imputation_flags_are_rejected(cleaned, tmp_path, value):
    cleaned['imputed_ni_pct'] = cleaned['imputed_ni_pct'].astype(object)
    cleaned.loc[0, 'imputed_ni_pct'] = value
    path = tmp_path / 'p1_processed.csv'
    cleaned.to_csv(path, index=False)
    with pytest.raises(ValueError, match='결측 이력 플래그'):
        load_p1_output(path)
    with pytest.raises(ValueError, match='결측 이력 플래그'):
        prepare_processed(cleaned)
