"""Features del baseline clásico B1: HOG sobre la imagen en gris de 64x64.

No importa torch: joblib lanza procesos nuevos que importan este módulo, y así arrancan
rápido y con poca memoria.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

HOG_DEFAULTS: dict = {
    "image_size": 64,
    "orientations": 9,
    "pixels_per_cell": 8,
    "cells_per_block": 2,
}


def hog_one(path: str, image_size: int = 64, orientations: int = 9,
            pixels_per_cell: int = 8, cells_per_block: int = 2) -> np.ndarray:
    """HOG de una imagen: gris → redimensionar a ``image_size`` → ``skimage.feature.hog``."""
    from PIL import Image
    from skimage.feature import hog

    with Image.open(path) as im:
        gray = im.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    feats = hog(
        arr,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feats.astype(np.float32)


def _hog_chunk(paths: Sequence[str], params: dict) -> np.ndarray:
    return np.stack([hog_one(p, **params) for p in paths])


def hog_features(paths: Sequence[str], params: dict | None = None, n_jobs: int = -1,
                 chunk_size: int = 512) -> np.ndarray:
    """Matriz de features HOG ``[N, D]`` (float32) extraída en paralelo con joblib."""
    from joblib import Parallel, delayed

    prm = {**HOG_DEFAULTS, **(params or {})}
    paths = list(paths)
    chunks = [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]
    parts = Parallel(n_jobs=n_jobs)(delayed(_hog_chunk)(c, prm) for c in chunks)
    return np.concatenate(parts, axis=0)
