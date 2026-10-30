import pytest

from module3_optimization.config import load_config

# 불순물 합 3% (A급 수준), 원료 100kg (실제 성공 배치 95~117kg) — 확률 제약에서도 달성 가능
FEED_A = {
    "batch_id": "TEST-A",
    "ni_pct": 25.0, "co_pct": 10.0, "li_pct": 4.0, "mn_pct": 8.0,
    "cu_pct": 1.0, "fe_pct": 0.8, "al_pct": 1.2,
    "feed_kg": 100.0,
}  # fmt: skip

# 불순물 합 16% (C급 수준) — 가짜 모델에서 어떤 조건으로도 달성 불가
FEED_C = {
    "batch_id": "TEST-C",
    "ni_pct": 10.0, "co_pct": 3.0, "li_pct": 3.0, "mn_pct": 12.0,
    "cu_pct": 5.0, "fe_pct": 5.0, "al_pct": 6.0,
    "feed_kg": 150.0,
}  # fmt: skip


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def feed_a():
    return dict(FEED_A)


@pytest.fixture
def feed_c():
    return dict(FEED_C)
