"""Arquitecturas: mAlexNet (desde cero), modelos de torchvision con fine-tuning y features HOG.

- ``malexnet``: CNN pequeña del paper de Amato et al. (2017), sin preentrenamiento.
- ``mobilenet_v3_small``, ``mobilenet_v3_large``, ``efficientnet_b0``, ``resnet18``,
  ``resnet50``: pesos de ImageNet de torchvision con la cabeza reemplazada a 2 clases.
- ``hog_features``: descriptor HOG del baseline clásico B1 (sin deep learning); vive en
  ``cupo.classical`` para que los procesos de joblib no importen torch.

Congelamiento (``freeze``):
- ``none``: todo entrenable;
- ``backbone``: solo la cabeza;
- ``partial``: los últimos ~30 % de los bloques del backbone + la cabeza.
"""

from __future__ import annotations

import io
import math
from typing import Literal

import torch
from torch import nn

from cupo.classical import HOG_DEFAULTS, hog_features  # noqa: F401  (reexportados)

FreezeMode = Literal["none", "backbone", "partial"]
FREEZE_MODES: tuple[str, ...] = ("none", "backbone", "partial")
TORCHVISION_MODELS: tuple[str, ...] = (
    "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18", "resnet50",
)
MODEL_NAMES: tuple[str, ...] = ("malexnet", *TORCHVISION_MODELS)
PARTIAL_FRACTION = 0.3  # fracción de bloques finales que se entrenan con freeze='partial'


# ---------------------------------------------------------------------------
# mAlexNet
# ---------------------------------------------------------------------------

class FP32LocalResponseNorm(nn.Module):
    """LRN que siempre se calcula en fp32 (también con AMP) y devuelve el dtype de entrada.

    En fp16 la suma de cuadrados de la LRN puede desbordarse o perder precisión.
    """

    def __init__(self, size: int = 5) -> None:
        super().__init__()
        self.lrn = nn.LocalResponseNorm(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.lrn(x.float()).to(x.dtype)


class MAlexNet(nn.Module):
    """mAlexNet (Amato et al., 2017): 3 convoluciones + 2 capas densas.

    conv1 16@11x11/4 → ReLU → LRN → MaxPool(3,2)      (la LRN corre en fp32)
    conv2 20@5x5     → ReLU → LRN → MaxPool(3,2)
    conv3 30@3x3     → ReLU → MaxPool(3,2)
    fc 48 → ReLU → fc num_classes

    Los MaxPool usan ``ceil_mode=True`` (como el pooling de Caffe del original); así la
    red también acepta entradas de 128x128.
    """

    def __init__(self, num_classes: int = 2, input_size: int = 224) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=11, stride=4),
                nn.ReLU(inplace=True),
                FP32LocalResponseNorm(5),
                nn.MaxPool2d(3, 2, ceil_mode=True),
            ),
            nn.Sequential(
                nn.Conv2d(16, 20, kernel_size=5),
                nn.ReLU(inplace=True),
                FP32LocalResponseNorm(5),
                nn.MaxPool2d(3, 2, ceil_mode=True),
            ),
            nn.Sequential(
                nn.Conv2d(20, 30, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, ceil_mode=True),
            ),
        )
        flat_dim = self._flat_dim(input_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, num_classes),
        )

    @torch.no_grad()
    def _flat_dim(self, input_size: int) -> int:
        """Dimensión del flatten calculada con un tensor dummy."""
        dummy = torch.zeros(1, 3, input_size, input_size)
        return int(self.features(dummy).numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Modelos de torchvision
# ---------------------------------------------------------------------------

def _torchvision_model(name: str, pretrained: bool) -> nn.Module:
    """Crea el modelo de torchvision con los pesos de ImageNet (o sin pesos)."""
    from torchvision import models as tvm

    builders = {
        "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights),
        "mobilenet_v3_large": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights),
        "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights),
        "resnet18": (tvm.resnet18, tvm.ResNet18_Weights),
        "resnet50": (tvm.resnet50, tvm.ResNet50_Weights),
    }
    builder, weights_enum = builders[name]
    return builder(weights=weights_enum.DEFAULT if pretrained else None)


def _replace_head(model: nn.Module, name: str, num_classes: int) -> None:
    """Reemplaza la última capa lineal por una de ``num_classes`` salidas."""
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, num_classes)


def head_module(model: nn.Module, name: str) -> nn.Module:
    """Cabeza del modelo (lo que se entrena con ``freeze='backbone'``)."""
    if name.startswith("resnet"):
        return model.fc
    return model.classifier


def backbone_blocks(model: nn.Module, name: str) -> list[nn.Module]:
    """Bloques del backbone en orden, de la entrada a la salida."""
    if name.startswith("resnet"):
        stem = nn.ModuleList([model.conv1, model.bn1])
        blocks = [b for layer in (model.layer1, model.layer2, model.layer3, model.layer4) for b in layer]
        return [stem, *blocks]
    # malexnet, mobilenet_v3 y efficientnet guardan sus bloques en ``features``
    return list(model.features)


def apply_freeze(model: nn.Module, name: str, freeze: str) -> None:
    """Marca qué parámetros se entrenan según ``freeze``."""
    if freeze not in FREEZE_MODES:
        raise ValueError(f"freeze debe ser uno de {FREEZE_MODES}, se recibió {freeze!r}")
    if freeze == "none":
        for prm in model.parameters():
            prm.requires_grad = True
        return
    for prm in model.parameters():
        prm.requires_grad = False
    trainable: list[nn.Module] = [head_module(model, name)]
    if freeze == "partial":
        blocks = backbone_blocks(model, name)
        n_last = max(1, math.ceil(PARTIAL_FRACTION * len(blocks)))
        trainable += blocks[-n_last:]
    for module in trainable:
        for prm in module.parameters():
            prm.requires_grad = True


def build_model(
    name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze: str = "none",
    input_size: int = 224,
) -> nn.Module:
    """Construye el modelo ``name`` con la cabeza a ``num_classes`` y el congelamiento pedido.

    ``input_size`` solo afecta a mAlexNet (dimensión del flatten). mAlexNet ignora
    ``pretrained`` porque no tiene pesos preentrenados.
    """
    if name == "malexnet":
        model: nn.Module = MAlexNet(num_classes=num_classes, input_size=input_size)
    elif name in TORCHVISION_MODELS:
        model = _torchvision_model(name, pretrained)
        _replace_head(model, name, num_classes)
    else:
        raise ValueError(f"Modelo desconocido: {name!r}. Opciones: {MODEL_NAMES}")
    apply_freeze(model, name, freeze)
    return model


def set_train_mode(model: nn.Module) -> nn.Module:
    """``model.train()``, pero las BatchNorm congeladas se quedan en modo evaluación.

    Así un backbone congelado no cambia sus estadísticas de ImageNet.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(prm.requires_grad for prm in module.parameters()):
                module.eval()
    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    """(parámetros totales, parámetros entrenables)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def model_size_mb(model: nn.Module) -> float:
    """Tamaño en MB del ``state_dict`` serializado."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1024**2
