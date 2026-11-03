import math

import numpy as np
import pandas as pd
import pytest

from module3_optimization import constraints as C
from module3_optimization.config import config_hash, load_config
from module3_optimization.predictors import MockAnalyticPredictor, RecoveryModelAdapter
from module3_optimization.problem import InputError, Problem, Recipe, validate_feed

# ── 제약 ───────────────────────────────────────────────


def test_bounded_includes_edges_and_measures_violation():
    ok, viol = C.bounded(pd.Series([399.0, 400.0, 1500.0, 1510.0]), 400, 1500)
    assert ok.tolist() == [False, True, True, False]
    assert viol.tolist() == [1.0, 0.0, 0.0, 10.0]


def test_bounded_none_side_is_ignored():
    ok, _ = C.bounded(pd.Series([-1e9, 1e9]), lo=None, hi=None)
    assert ok.all()


def test_round_to_unit_snaps_and_clips(cfg):
    U = pd.DataFrame(
        {
            "acid_l": [404.9, 1499.0, 2000.0],
            "temp_c": [40.4, 94.6, 10.0],
            "time_hr": [1.2, 7.8, 9.0],
        }
    )
    R = C.round_to_unit(U, cfg["decision_variables"])
    assert R["acid_l"].tolist() == [400.0, 1500.0, 1500.0]
    assert R["temp_c"].tolist() == [40.0, 95.0, 40.0]
    assert R["time_hr"].tolist() == [1.0, 8.0, 8.0]


# ── 예측기 ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "var, values",
    [("acid_l", [400, 800, 1200]), ("temp_c", [40, 70, 95]), ("time_hr", [1, 4, 8])],
)
def test_mock_is_monotone_in_each_decision_variable(feed_a, var, values):
    base = {"acid_l": 800.0, "temp_c": 80.0, "time_hr": 4.0, **feed_a}
    X = pd.DataFrame([{**base, var: v} for v in values])
    pred = MockAnalyticPredictor().predict(X)
    assert pred["ni_pred"].is_monotonic_increasing
    assert pred["co_pred"].is_monotonic_increasing
    assert list(pred.index) == list(X.index)


class _DummyModel:
    def predict(self, X):
        return np.column_stack([np.full(len(X), 96.0), np.full(len(X), 91.0)])


def test_adapter_converts_ndarray_and_fills_zero_std():
    X = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 11, 12])
    adapter = RecoveryModelAdapter(_DummyModel())
    out = adapter.predict(X)
    assert list(out.index) == [10, 11, 12]
    assert out["ni_pred"].eq(96.0).all() and out["co_std"].eq(0.0).all()
    assert not adapter.has_std


def test_adapter_rejects_wrong_shape():
    class Bad:
        def predict(self, X):
            return np.zeros(len(X))

    with pytest.raises(ValueError):
        RecoveryModelAdapter(Bad()).predict(pd.DataFrame({"a": [1, 2]}))


# ── 입력 검증 · 설정 ────────────────────────────────────


def test_validate_feed_rejects_missing_and_nan(feed_a):
    del feed_a["cu_pct"]
    feed_a["fe_pct"] = math.nan
    with pytest.raises(InputError) as exc:
        validate_feed(feed_a)
    assert exc.value.missing == ["cu_pct"]
    assert exc.value.invalid == ["fe_pct"]


def test_validate_feed_keeps_optional_keys(feed_a):
    assert validate_feed(feed_a)["batch_id"] == "TEST-A"


def test_config_hash_tracks_changes(cfg):
    assert config_hash(cfg) == config_hash(load_config())
    assert config_hash(cfg) != config_hash(
        load_config(overrides={"targets": {"ni_recovery_pct": 96.0}})
    )


# ── Problem.evaluate ───────────────────────────────────


def test_evaluate_single_row_equals_batch_row(cfg, feed_a):
    problem = Problem.from_config(feed_a, cfg)
    U = pd.DataFrame(
        {
            "acid_l": [500.0, 900.0, 1300.0],
            "temp_c": [60.0, 80.0, 95.0],
            "time_hr": [2.0, 4.0, 8.0],
        }
    )
    batch = problem.evaluate(U)
    for i in range(len(U)):
        single = problem.evaluate(U.iloc[[i]])
        pd.testing.assert_series_equal(single.iloc[0], batch.iloc[i], check_names=False)


def test_evaluate_survives_feature_fn_that_resets_index(cfg, feed_a):
    """P2 add_features 가 index 를 초기화해도 예측이 올바른 행에 붙어야 한다."""
    U = pd.DataFrame(
        {"acid_l": [500.0, 1500.0], "temp_c": [40.0, 95.0], "time_hr": [1.0, 8.0]},
        index=[7, 3],
    )
    plain = Problem.from_config(feed_a, cfg).evaluate(U)
    resetting = Problem.from_config(feed_a, cfg)
    resetting.feature_fn = lambda X: X.reset_index(drop=True)
    out = resetting.evaluate(U)
    assert list(out.index) == [7, 3]
    assert not out["ni_pred"].isna().any()
    pd.testing.assert_frame_equal(out, plain)


def test_evaluate_feasible_requires_every_constraint(cfg, feed_a):
    problem = Problem.from_config(feed_a, cfg)
    out = problem.evaluate(
        pd.DataFrame(
            {"acid_l": [1500.0, 1600.0], "temp_c": [95.0, 95.0], "time_hr": [8.0, 8.0]}
        )
    )
    assert out["feasible"].tolist() == [True, False]
    assert not out.loc[1, "ok_box_acid_l"] and out.loc[1, "violation"] > 0


def test_chance_constraint_uses_lower_bound(cfg, feed_a):
    problem = Problem.from_config(feed_a, cfg)
    out = problem.evaluate(
        pd.DataFrame({"acid_l": [1000.0], "temp_c": [90.0], "time_hr": [6.0]})
    )
    z = cfg["uncertainty"]["chance_z"]
    assert out.loc[0, "ni_judged"] == pytest.approx(
        out.loc[0, "ni_pred"] - z * out.loc[0, "ni_std"]
    )


def test_target_override_only_replaces_given_key(cfg, feed_a):
    problem = Problem.from_config(feed_a, cfg, target={"ni_recovery_pct": 96.0})
    assert problem.target == {"ni_recovery_pct": 96.0, "co_recovery_pct": 90.0}


def test_ratio_constraint_applies_when_configured(feed_a):
    feed_a["feed_kg"] = 150.0
    cfg = load_config(
        overrides={"ratio_constraints": {"acid_per_feed": {"min": 3.0, "max": 8.0}}}
    )
    problem = Problem.from_config(feed_a, cfg)  # feed_kg = 150 → 450 ~ 1200 L
    out = problem.evaluate(
        pd.DataFrame(
            {
                "acid_l": [400.0, 600.0, 1300.0],
                "temp_c": [95.0] * 3,
                "time_hr": [8.0] * 3,
            }
        )
    )
    assert out["ok_ratio_acid_per_feed"].tolist() == [False, True, False]


# ── Recipe ─────────────────────────────────────────────


def test_recipe_rejects_unknown_status():
    with pytest.raises(ValueError):
        Recipe(status="maybe", target={})


def test_recipe_json_has_schema_keys():
    d = Recipe(
        status="out_of_domain",
        target={"ni_recovery_pct": 95.0, "co_recovery_pct": 90.0},
    ).to_dict()
    assert list(d)[:3] == ["schema_version", "status", "batch_id"]
    assert d["recipe"] is None and d["prediction"] is None
