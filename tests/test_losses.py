"""Pruebas de losses.py: peso de la clase OCUPADO y label smoothing (valores calculados a mano)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from cupo.losses import BUSY, build_loss  # noqa: E402


def _softplus(x: float) -> float:
    return math.log1p(math.exp(x))


# Dos ejemplos: libre con logits [2, 0] y ocupado con logits [0, 0]
LOGITS = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
TARGET = torch.tensor([0, 1])
CE_FREE = _softplus(-2.0)    # -log(sigmoid(2)) = log(1 + e^-2) = 0.126928
CE_BUSY = math.log(2.0)      # 0.693147


def test_busy_index_is_one():
    assert BUSY == 1


def test_default_loss_is_plain_cross_entropy():
    crit = build_loss({})
    assert crit.weight is None and crit.label_smoothing == 0.0
    assert float(crit(LOGITS, TARGET)) == pytest.approx((CE_FREE + CE_BUSY) / 2, rel=1e-6)
    assert float(crit(LOGITS, TARGET)) == pytest.approx(0.4100377, rel=1e-6)


def test_cost_weight_busy_weights_class_one_by_hand():
    """Media ponderada de PyTorch: (1·0.126928 + 3·0.693147) / (1 + 3) = 0.551592."""
    crit = build_loss({"cost_weight_busy": 3.0})
    assert crit.weight.tolist() == [1.0, 3.0]
    expected = (1 * CE_FREE + 3 * CE_BUSY) / 4
    assert expected == pytest.approx(0.551592, abs=1e-6)
    assert float(crit(LOGITS, TARGET)) == pytest.approx(expected, rel=1e-6)


def test_cost_weight_penalizes_busy_errors_more():
    """El error ocupado→libre pesa más que libre→ocupado con el mismo margen."""
    crit = build_loss({"cost_weight_busy": 4.0})
    busy_as_free = torch.tensor([[1.0, 0.0]]), torch.tensor([1])
    free_as_busy = torch.tensor([[0.0, 1.0]]), torch.tensor([0])
    # Con un solo ejemplo la media ponderada se normaliza; se comparan sumas
    crit_sum = torch.nn.CrossEntropyLoss(weight=crit.weight, reduction="sum")
    l_bf = float(crit_sum(*busy_as_free))
    l_fb = float(crit_sum(*free_as_busy))
    assert l_bf == pytest.approx(4 * l_fb, rel=1e-6)
    assert l_fb == pytest.approx(_softplus(1.0), rel=1e-6)


def test_label_smoothing_by_hand():
    """ε = 0.1, 2 clases, y = 0, logits [2, 0]:
    pérdida = -(1 - ε + ε/2)·log σ(2) - (ε/2)·log σ(-2) = 0.95·0.126928 + 0.05·2.126928 = 0.226928.
    """
    crit = build_loss({"label_smoothing": 0.1})
    x, y = torch.tensor([[2.0, 0.0]]), torch.tensor([0])
    expected = 0.95 * _softplus(-2.0) + 0.05 * _softplus(2.0)
    assert expected == pytest.approx(0.226928, abs=1e-6)
    assert float(crit(x, y)) == pytest.approx(expected, rel=1e-6)


def test_label_smoothing_and_weight_together():
    crit = build_loss({"label_smoothing": 0.2, "cost_weight_busy": 2.0})
    ref = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]), label_smoothing=0.2)
    assert float(crit(LOGITS, TARGET)) == pytest.approx(float(ref(LOGITS, TARGET)), rel=1e-6)


def test_accepts_full_config_with_loss_section():
    crit = build_loss({"name": "x", "loss": {"cost_weight_busy": 3.0, "label_smoothing": 0.05}})
    assert crit.weight.tolist() == [1.0, 3.0]
    assert crit.label_smoothing == pytest.approx(0.05)


def test_none_config_is_default():
    crit = build_loss(None)
    assert crit.weight is None and crit.label_smoothing == 0.0


@pytest.mark.parametrize("bad", [{"cost_weight_busy": 0.5}, {"label_smoothing": -0.1},
                                 {"label_smoothing": 1.0}])
def test_invalid_values_raise(bad):
    with pytest.raises(ValueError):
        build_loss(bad)


def test_k4_config_uses_busy_weight_three():
    from cupo.runs import load_yaml
    from cupo import paths

    cfg = load_yaml(paths.configs_dir() / "k4_mnv3_ft_cost.yaml")
    crit = build_loss(cfg)
    assert crit.weight.tolist() == [1.0, 3.0]
