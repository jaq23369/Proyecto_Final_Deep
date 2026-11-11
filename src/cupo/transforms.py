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


def build_transforms(input_size: int, train: bool, augment: bool):
    """Transformación de torchvision para entrenamiento o evaluación.

    El aumento de datos solo se aplica si ``train and augment``.
    """
    # Import local: la app (ONNX) usa este módulo sin tener torchvision
    from torchvision import transforms as T

    ops: list = [T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BILINEAR)]
    use_aug = train and augment
    if use_aug:
        ops += [
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.02),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
            T.RandomHorizontalFlip(),
        ]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
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
