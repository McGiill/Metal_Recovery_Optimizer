"""Module 2 테스트. 소유자: P3

실행: python -m pytest module2_prediction/tests -q   (data/processed/batches.csv 필요)
"""

import numpy as np
import pandas as pd
import pytest

from module2_prediction import config as C
from module2_prediction.data import check_against_raw, load_module1, split_dev_test
from module2_prediction.features import FeatureBuilder, GradeMedianImputer
from module2_prediction.inference import mix_materials, predict_batch
from module2_prediction.model import RecoveryModel

pytestmark = pytest.mark.skipif(not C.M1_BATCHES.exists(), reason="Module 1 출력 없음")


@pytest.fixture(scope="module")
def df():
    return load_module1()


def test_nan_restored_matches_raw(df):
    check_against_raw(df)
    assert int(df[C.FEATURE_COLS].isna().sum().sum()) == 291
    assert int(df["process_complete"].sum()) == 345


def test_split_is_time_ordered(df):
    dev, test = split_dev_test(df)
    assert dev["batch_seq"].max() < test["batch_seq"].min()
    assert len(dev) + len(test) == len(df)


def test_imputer_uses_train_only():
    train = pd.DataFrame({"grade": ["A", "A", "B"], "x": [1.0, 3.0, 10.0]})
    other = pd.DataFrame({"grade": ["A", "B", "Z"], "x": [np.nan, np.nan, np.nan]})
    imp = GradeMedianImputer(columns=("x",)).fit(train)
    with pytest.warns(UserWarning):
        out = imp.transform(other)
    assert out["x"].tolist() == [2.0, 10.0, 3.0]  # 처음 보는 등급 → 학습 전역 중앙값
    assert other["x"].isna().all()  # 원본 불변


def test_policy_b_keeps_derived_nan(df):
    row = df[df["time_hr"].isna()].head(1)
    fb = FeatureBuilder(derived=("acid_time_interaction",)).fit(row)
    assert np.isnan(fb.transform(row)["acid_time_interaction"].iloc[0])


def test_predict_contract_and_policy_c_rejects(df):
    dev, test = split_dev_test(df)
    model = RecoveryModel("ridge", "C", "F4").fit(dev, dev[C.TARGETS])
    pred = model.predict(test)
    assert pred.shape == (len(test), 2)
    incomplete = ~test["process_complete"].to_numpy()
    assert np.isnan(pred[incomplete]).all() and not np.isnan(pred[~incomplete]).any()
    # grade 없이 성분이 모두 있는 입력도 예측 (Module 3 호출 형태)
    full = test[test["process_complete"] & test["composition_complete"]][C.FEATURE_COLS]
    assert not np.isnan(model.predict(full)).any()


def test_policy_b_requires_nan_capable_model():
    with pytest.raises(ValueError):
        RecoveryModel("ridge", "B", "F0")


def test_predict_batch_status(df):
    dev, _ = split_dev_test(df)
    model = RecoveryModel("ridge", "C", "F0").fit(dev, dev[C.TARGETS])
    cand = dev[dev["process_complete"] & dev["composition_complete"]].head(3)[["grade"] + C.FEATURE_COLS].copy()
    cand.iloc[1, cand.columns.get_loc("temp_c")] = np.nan
    cand.iloc[2, cand.columns.get_loc("acid_l")] = 5000
    out = predict_batch(model, cand)
    assert list(out.index) == list(cand.index)
    assert out["status"].tolist() == ["ok", "invalid_input", "out_of_domain"]
    assert out.iloc[1][["ni_recovery_pct_pred", "co_recovery_pct_pred"]].isna().all()


def test_mix_materials_weighted_average():
    a = {c: 10.0 for c in C.COMPOSITION} | {"grade": "A"}
    b = {c: 20.0 for c in C.COMPOSITION} | {"grade": "C", "ni_pct": None}
    fill = {"C": {c: 30.0 for c in C.COMPOSITION}}
    mix = mix_materials(a, b, 0.7, 150.0, composition_fill=fill)
    assert mix["co_pct"] == pytest.approx(13.0)
    assert mix["ni_pct"] == pytest.approx(0.7 * 10 + 0.3 * 30)
    assert mix["grade"] is None and mix["feed_kg"] == 150.0
