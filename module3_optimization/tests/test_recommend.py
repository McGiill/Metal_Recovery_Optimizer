import json

import pandas as pd
import pytest

from module3_optimization import report
from module3_optimization.config import load_config
from module3_optimization.domain import build_domains
from module3_optimization.optimizers import GridSearch
from module3_optimization.problem import InputError, Problem
from module3_optimization.recommend import feed_from_row, recommend
from module3_optimization.tests.test_domain import synthetic_train


@pytest.fixture
def cfg_nodomain():
    return load_config(overrides={"domain": {"enabled": False}})


@pytest.fixture
def domains(cfg):
    return build_domains(cfg, synthetic_train())


# ── 상태별 결과 ────────────────────────────────────────


@pytest.mark.parametrize("chance_enabled", [True, False])
@pytest.mark.parametrize(
    "bad_column, bad_value", [("ni_pred", float("nan")), ("co_std", float("inf"))]
)
@pytest.mark.parametrize("scenario", ["mixed", "all_invalid", "fallback"])
def test_recommend_never_returns_invalid_predictions(
    cfg_nodomain, feed_a, monkeypatch, chance_enabled, bad_column, bad_value, scenario
):
    class Predictor:
        def predict(self, X):
            meets_target = scenario != "fallback"
            out = pd.DataFrame(
                {
                    "ni_pred": 96.0 if meets_target else 94.0,
                    "co_pred": 91.0 if meets_target else 89.0,
                    "ni_std": 0.25,
                    "co_std": 0.25,
                },
                index=X.index,
            )
            bad = (
                pd.Series(True, index=X.index)
                if scenario == "all_invalid"
                else X["acid_l"].eq(400)
            )
            out.loc[bad, bad_column] = bad_value
            return out

    monkeypatch.setattr(
        "module3_optimization.recommend.build_predictor", lambda cfg: Predictor()
    )
    cfg_nodomain["uncertainty"]["use_chance_constraint"] = chance_enabled
    cfg_nodomain["decision_variables"] = {
        "acid_l": {"min": 400, "max": 420, "unit": 10},
        "temp_c": {"min": 90, "max": 90, "unit": 1},
        "time_hr": {"min": 6, "max": 6, "unit": 0.5},
    }
    result = recommend(feed_a, cfg=cfg_nodomain)
    if scenario == "all_invalid":
        assert result.status == "infeasible"
        assert result.recipe is None and result.prediction is None
        assert result.shortfall is None
    else:
        assert result.status == ("ok" if scenario == "mixed" else "infeasible")
        assert result.recipe["acid_l"] == 410
        assert result.prediction["ni"]["mean"] == (
            96.0 if scenario == "mixed" else 94.0
        )
        if scenario == "fallback":
            assert result.shortfall["ni"] > 0 and result.shortfall["co"] > 0
    # JSON 응답에 NaN/Infinity 가 남으면 표준 JSON 직렬화가 실패한다.
    json.dumps(result.to_dict(), allow_nan=False)


def test_ok_recipe_meets_target_and_constraints(cfg_nodomain, feed_a):
    r = recommend(feed_a, cfg=cfg_nodomain)
    assert r.status == "ok" and r.shortfall is None
    z = cfg_nodomain["uncertainty"]["chance_z"]
    # 구간 표시는 1.645σ, 판정은 1.28σ — 판정 하한이 목표 이상
    ni = r.prediction["ni"]
    sigma = (ni["high"] - ni["mean"]) / cfg_nodomain["uncertainty"]["interval_z"]
    assert ni["mean"] - z * sigma >= 95.0 - 1e-6
    spec = cfg_nodomain["decision_variables"]
    for var in ("acid_l", "temp_c", "time_hr"):
        assert spec[var]["min"] <= r.recipe[var] <= spec[var]["max"]
    assert r.recipe["acid_per_feed"] == pytest.approx(
        r.recipe["acid_l"] / 100.0, abs=0.01
    )


def test_infeasible_returns_closest_with_shortfall(cfg_nodomain, feed_c):
    r = recommend(feed_c, cfg=cfg_nodomain)
    assert r.status == "infeasible" and r.confidence == "low"
    assert r.recipe is not None and r.shortfall["co"] > 0
    assert "블렌딩" in r.message  # 불순물 16%


def test_impossible_target_is_infeasible(cfg_nodomain, feed_a):
    r = recommend(
        feed_a, target={"ni_recovery_pct": 120.0}, cfg=cfg_nodomain, algo="grid_coarse"
    )
    assert r.status == "infeasible"


def test_extreme_feed_is_out_of_domain(cfg, feed_a, domains):
    process, feed_dom = domains
    feed_a.update(ni_pct=60.0, co_pct=40.0, feed_kg=900.0)
    r = recommend(feed_a, cfg=cfg, domain=process, feed_domain=feed_dom)
    assert r.status == "out_of_domain" and r.recipe is None and r.prediction is None


def test_domain_adds_similar_batches(cfg, feed_a, domains):
    process, feed_dom = domains
    r = recommend(
        feed_a, cfg=cfg, domain=process, feed_domain=feed_dom, algo="grid_coarse"
    )
    assert r.status in ("ok", "infeasible")
    assert r.similar_batches is not None and r.domain_distance is not None


def test_missing_feed_raises(cfg_nodomain, feed_a):
    feed_a["co_pct"] = None
    with pytest.raises(InputError):
        recommend(feed_a, cfg=cfg_nodomain)


def test_interval_is_clipped_to_100(cfg_nodomain, feed_a):
    r = recommend(feed_a, cfg=cfg_nodomain)
    assert all(0 <= v <= 100 for m in r.prediction.values() for v in m.values())


def test_deterministic(cfg_nodomain, feed_a):
    a = recommend(feed_a, cfg=cfg_nodomain, algo="grid_coarse").to_dict()
    b = recommend(feed_a, cfg=cfg_nodomain, algo="grid_coarse").to_dict()
    for d in (a, b):
        d["meta"].pop("elapsed_ms")
        d["meta"].pop("generated_at")
    assert a == b


# ── 정답 검증: 해석 모델은 단조라 최소 황산량을 이분 탐색으로 정확히 구할 수 있다 ──


def true_min_acid(problem, temp_c, time_hr, lo=400.0, hi=1500.0, tol=0.01):
    def ok(acid):
        U = pd.DataFrame({"acid_l": [acid], "temp_c": [temp_c], "time_hr": [time_hr]})
        return bool(problem.evaluate(U)["feasible"].iloc[0])

    if not ok(hi):
        return None
    if ok(lo):
        return lo
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def test_grid_matches_bisection_optimum(cfg_nodomain, feed_a):
    problem = Problem.from_config(feed_a, cfg_nodomain)
    unit = cfg_nodomain["decision_variables"]["acid_l"]["unit"]
    # 비교 대상 (T, t) 를 성긴 격자로 — 연속 최적값 중 최소가 전역 최적
    answers = [
        true_min_acid(problem, T, t) for T in range(40, 96, 5) for t in (1, 2, 4, 6, 8)
    ]
    best_true = min(a for a in answers if a is not None)
    grid_best = GridSearch().run(problem).best["acid_l"]
    assert best_true <= grid_best < best_true + unit + 1e-6


# ── 출력 ───────────────────────────────────────────────


def test_card_and_json_for_each_status(cfg, cfg_nodomain, feed_a, feed_c, domains):
    process, feed_dom = domains
    extreme = {**feed_a, "ni_pct": 60.0, "co_pct": 40.0, "feed_kg": 900.0}
    recipes = [
        recommend(feed_a, cfg=cfg_nodomain),
        recommend(feed_c, cfg=cfg_nodomain),
        recommend(extreme, cfg=cfg, domain=process, feed_domain=feed_dom),
    ]
    titles = ["[추천 레시피]", "[목표 달성 불가]", "[추천 불가"]
    for r, title in zip(recipes, titles):
        card = report.format_card(r)
        assert card.startswith(title)
        # 가짜 모델 경고는 predictor.type=mock_analytic 일 때만 붙는다.
        # 실제 모델(real)에서는 나오지 않으므로 카드 본문만 검증한다.
        assert "배치" in card or "목표" in card
        assert json.loads(r.to_json())["status"] == r.status


def test_csv_roundtrip(tmp_path, cfg_nodomain, feed_a, feed_c):
    recipes = [
        recommend(f, cfg=cfg_nodomain, algo="grid_coarse") for f in (feed_a, feed_c)
    ]
    path = tmp_path / "out" / "recipes.csv"
    report.write_csv(recipes, path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    assert df["status"].tolist() == ["ok", "infeasible"]
    assert {"acid_l", "ni_mean", "shortfall_co", "model_version"} <= set(df.columns)


def test_feed_from_row_turns_nan_into_none(feed_a):
    row = pd.Series({**feed_a, "co_pct": float("nan"), "grade": "A"})
    feed = feed_from_row(row)
    assert feed["co_pct"] is None and feed["grade"] == "A"
