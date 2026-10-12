import numpy as np
import pandas as pd

from module3_optimization.optimizers import (
    GridSearch,
    get_optimizer,
    make_grid,
    select_best,
    select_closest,
)
from module3_optimization.problem import Problem


def test_grid_sizes(cfg):
    assert len(make_grid(cfg["decision_variables"], "unit")) == 111 * 56 * 15
    assert len(make_grid(cfg["decision_variables"], "coarse_step")) == 23 * 12 * 15


def test_grid_finds_feasible_recipe_on_field_units(cfg, feed_a):
    result = GridSearch().run(Problem.from_config(feed_a, cfg))
    best = result.best
    assert result.found and best["feasible"]
    assert best["ni_judged"] >= 95.0 and best["co_judged"] >= 90.0
    for var, spec in cfg["decision_variables"].items():
        steps = (best[var] - spec["min"]) / spec["unit"]
        assert spec["min"] <= best[var] <= spec["max"]
        assert np.isclose(steps, round(steps))


def test_grid_best_is_cheapest_feasible(cfg, feed_a):
    result = GridSearch().run(Problem.from_config(feed_a, cfg))
    assert (
        result.best["cost"]
        == result.history.loc[result.history["feasible"], "cost"].min()
    )


def test_grid_reports_infeasible_with_closest_candidate(cfg, feed_c):
    result = GridSearch().run(Problem.from_config(feed_c, cfg))
    assert not result.found
    closest = select_closest(result.history)
    assert closest is not None and closest["hard_ok"]
    assert closest["shortfall_total"] == result.history["shortfall_total"].min()


def test_impossible_target_is_infeasible(cfg, feed_a):
    result = GridSearch("coarse_step").run(
        Problem.from_config(feed_a, cfg, target={"ni_recovery_pct": 120.0})
    )
    assert not result.found


def test_tie_break_order():
    history = pd.DataFrame(
        {
            "feasible": [True, True, True, False],
            "cost": [500.0, 500.0, 500.0, 100.0],
            "uncertainty": [2.0, 1.0, 1.0, 0.0],
            "temp_c": [90.0, 95.0, 85.0, 40.0],
            "time_hr": [4.0, 4.0, 4.0, 1.0],
        }
    )
    best = select_best(
        history, ["cost", "uncertainty", "domain_distance", "temp_c", "time_hr"]
    )
    assert (
        best.name == 2
    )  # 비용 동점 → 불확실성 → (domain_distance 없음, 건너뜀) → 온도


def test_registry():
    assert get_optimizer("grid_coarse").step_key == "coarse_step"
