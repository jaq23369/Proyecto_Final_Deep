"""Función de pérdida: CrossEntropy con label smoothing y peso de costo para OCUPADO."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

BUSY = 1  # índice de la clase ocupado (0 = libre)


def build_loss(cfg: Mapping[str, Any] | None = None) -> nn.CrossEntropyLoss:
    """CrossEntropy a partir de la sección ``loss`` de la config (o la config completa).

    - ``label_smoothing`` (≥ 0): suavizado de etiquetas.
    - ``cost_weight_busy`` (≥ 1): peso de la clase OCUPADO. Con un peso mayor que 1,
      equivocarse en un espacio ocupado (mostrarlo libre) cuesta más en la pérdida.
    """
    cfg = dict(cfg or {})
    if "loss" in cfg and isinstance(cfg["loss"], Mapping):
        cfg = dict(cfg["loss"])
    smoothing = float(cfg.get("label_smoothing", 0.0))
    w_busy = float(cfg.get("cost_weight_busy", 1.0))
    if not 0.0 <= smoothing < 1.0:
        raise ValueError(f"label_smoothing debe estar en [0, 1), se recibió {smoothing}")
    if w_busy < 1.0:
        raise ValueError(f"cost_weight_busy debe ser >= 1, se recibió {w_busy}")
    weight = None
    if w_busy != 1.0:
        weight = torch.ones(2)
        weight[BUSY] = w_busy
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)
