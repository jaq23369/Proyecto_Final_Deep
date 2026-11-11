"""Preprocesamiento y aumento de datos.

La normalización es la de ImageNet (fija), así no se calcula ninguna estadística
con los datos. ``preprocess_numpy`` replica el preprocesamiento de evaluación sin
torch, para la app con ONNX Runtime.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Desenfoque gaussiano pensado sobre el patch original (~150 px, no sobre 224).
# Equivale a sigma (0.1, 1.5) a 224 px escalado por 150/224 ≈ 0.67 → (0.07, 1.0).
# El kernel proporcional (5 · 0.67 ≈ 3.3) se redondea al siguiente impar, 5: con
# 3 el gaussiano quedaría truncado a 1 sigma y se parecería a un filtro de caja.
BLUR_KERNEL: int = 5
BLUR_SIGMA: tuple[float, float] = (0.07, 1.0)


def _augmentations(T) -> list:
    """Aumento geométrico y fotométrico sobre el patch original, antes del Resize.

    Trabajar a ~150 px en lugar de 224 reduce el costo por imagen; todos los
    parámetros son relativos al tamaño (traslación, escala) o no dependen de él,
    salvo el desenfoque, que se ajusta arriba.
    """
    return [
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.02),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.RandomApply([T.GaussianBlur(kernel_size=BLUR_KERNEL, sigma=BLUR_SIGMA)], p=0.2),
        T.RandomHorizontalFlip(),
    ]


def build_transforms(input_size: int, train: bool, augment: bool):
    """Transformación de torchvision para entrenamiento o evaluación.

    El aumento de datos solo se aplica si ``train and augment``. Sin aumento, el
    pipeline es Resize bilineal → ToTensor → Normalize (igual a ``preprocess_numpy``).
    """
    # Import local: la app (ONNX) usa este módulo sin tener torchvision
    from torchvision import transforms as T

    use_aug = train and augment
    ops: list = _augmentations(T) if use_aug else []
    ops += [
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if use_aug:
        # Rectángulos pequeños borrados para simular oclusiones (postes, peatones, ramas)
        ops.append(T.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0))
    return T.Compose(ops)


def preprocess_numpy(img: Image.Image, input_size: int) -> np.ndarray:
    """Preprocesamiento de evaluación sin torch → arreglo ``[1, 3, H, W]`` float32."""
    img = img.convert("RGB").resize((input_size, input_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return arr.transpose(2, 0, 1)[None].astype(np.float32, copy=False)
