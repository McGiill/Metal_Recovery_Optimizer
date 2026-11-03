import numpy as np
import pandas as pd
import pytest

from module3_optimization.domain import FEED_COLS, DomainChecker, build_domains


def synthetic_train(n=300, seed=0):
    """원본과 비슷한 범위의 균등 난수 학습 데이터 (원본 파일 없이 테스트하기 위함)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "acid_l": rng.uniform(400, 1500, n),
            "temp_c": rng.uniform(40, 95, n),
            "time_hr": rng.uniform(1, 8, n),
            "ni_pct": rng.uniform(10, 28, n),
            "co_pct": rng.uniform(3, 13, n),
            "li_pct": rng.uniform(2, 6, n),
            "mn_pct": rng.uniform(5, 16, n),
            "cu_pct": rng.uniform(0.3, 5, n),
            "fe_pct": rng.uniform(0.2, 5, n),
            "al_pct": rng.uniform(0.5, 6, n),
            "feed_kg": rng.uniform(80, 200, n),
        }
    )


@pytest.fixture
def train():
    return synthetic_train()


def test_threshold_is_training_percentile(train):
    checker = DomainChecker().fit(
        train, ["acid_l", "temp_c", "time_hr"], k=5, percentile=95
    )
    assert np.mean(checker.train_distance_ <= checker.threshold) == pytest.approx(
        0.95, abs=0.01
    )


def test_far_point_is_out_of_domain(train):
    checker = DomainChecker().fit(train, ["acid_l", "temp_c", "time_hr"])
    X = pd.DataFrame(
        {"acid_l": [950.0, 5000.0], "temp_c": [67.0, 300.0], "time_hr": [4.5, 40.0]}
    )
    assert checker.in_domain(X).tolist() == [True, False]
    counts = checker.similar_count(X)
    assert counts[0] > 0 and counts[1] == 0


def test_fit_ignores_rows_with_nan(train):
    train.loc[:9, "acid_l"] = np.nan
    checker = DomainChecker().fit(train, ["acid_l", "temp_c"])
    assert checker.n_train == len(train) - 10


def test_distance_rejects_nan_query(train):
    checker = DomainChecker().fit(train, ["acid_l", "temp_c"])
    with pytest.raises(ValueError):
        checker.distance(pd.DataFrame({"acid_l": [np.nan], "temp_c": [60.0]}))


def test_build_domains_returns_process_and_feed(cfg, train):
    process, feed = build_domains(cfg, train)
    assert feed.cols == FEED_COLS
    assert set(FEED_COLS) < set(process.cols)
