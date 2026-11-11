"""Fixtures compartidas: escenarios de valor y datos sintéticos con semilla fija."""

from __future__ import annotations

import numpy as np
import pytest

from cupo import value_model as vm


def make_rates(q_OL: float, q_OV: float, q_LL: float, q_LV: float) -> dict[str, float]:
    """Seis tasas de tres estados; q_OO y q_LO completan cada clase a 1."""
    return {
        "q_OL": q_OL,
        "q_OV": q_OV,
        "q_OO": 1.0 - q_OL - q_OV,
        "q_LL": q_LL,
        "q_LV": q_LV,
        "q_LO": 1.0 - q_LL - q_LV,
    }


def synthetic_scores(n: int, sep: float, noise: float, seed: int, busy_frac: float = 0.7):
    """Etiquetas (1 = ocupado) y P(ocupado) de un clasificador sintético."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, busy_frac, n)
    logit = np.where(y == 1, sep, -sep) + rng.normal(0.0, noise, n)
    p = 1.0 / (1.0 + np.exp(-logit))
    return y, p


@pytest.fixture(scope="session")
def com() -> dict:
    return vm.load_scenario("comercial")


@pytest.fixture(scope="session")
def uvg() -> dict:
    return vm.load_scenario("uvg")


@pytest.fixture(scope="session")
def good_scores():
    return synthetic_scores(20_000, sep=2.0, noise=1.5, seed=0)


@pytest.fixture(scope="session")
def weak_scores():
    return synthetic_scores(20_000, sep=0.8, noise=1.5, seed=1)
