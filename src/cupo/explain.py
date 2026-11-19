"""Explicabilidad del modelo final: selección de instancias, oclusión y pruebas de fidelidad.

Este módulo no entrena nada. Trabaja con las predicciones ya guardadas de un run
(``<split>_preds.npz``) y con el modelo ya entrenado.

- ``load_split_table``: tabla de un split con la probabilidad recalculada con la temperatura del run.
- ``select_instances``: elige las instancias locales que se explican (reglas fijas y semilla).
- ``occlusion_map``: mapa de oclusión (alternativa a Grad-CAM, sirve con cualquier modelo).
- ``deletion_fidelity``: prueba de fidelidad, tapa la zona que marca un mapa y mide la caída.
- ``center_share``: parte de la atención que cae en el centro del parche.
- ``hog_contribution_map``: explicación del baseline interpretable (HOG + regresión logística).
- ``stratified_sample``: muestra por clase y clima para la explicación global.

Grad-CAM vive en ``scripts/gradcam.py``; aquí se usa desde el notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from cupo.evaluate import probs_from_logits
from cupo.runs import LoadedRun, load_npz

SEED = 42

HIT_BUSY = "acierto ocupado (confianza alta)"
HIT_FREE = "acierto libre (confianza alta)"
ERR_BUSY_AS_FREE = "error ocupado→libre"
ERR_FREE_AS_BUSY = "error libre→ocupado"
NEAR_THRESHOLD = "cerca del umbral t_H"
RAIN_HIT = "acierto con lluvia"
OOD_CASE = "otro parqueo (OOD)"

# Una probabilidad cercana a 1 (o a 0) cuenta como confianza alta.
HIGH_BUSY = 0.999
HIGH_FREE = 0.002


@dataclass(frozen=True)
class Operating:
    """Punto de operación de un run: temperatura y umbrales del escenario de selección."""

    scenario: str
    t_L: float
    t_H: float
    temperature: float


def operating_point(run: LoadedRun) -> Operating:
    scenario = run.config.get("select", {}).get("scenario", "comercial")
    if scenario not in run.thresholds:
        scenario = next(iter(run.thresholds))
    thr = run.thresholds[scenario]
    return Operating(scenario, float(thr["t_L"]), float(thr["t_H"]), float(run.temperature))


def load_split_table(run: LoadedRun, split: str) -> pd.DataFrame:
    """Predicciones de un split (``val``, ``test`` u ``ood``) con ``p`` recalculada con la temperatura."""
    data = load_npz(run.run_dir / f"{split}_preds.npz")
    if "logits" in data:
        p = probs_from_logits(data["logits"], run.temperature)
    elif run.calibrated:
        raise ValueError(f"{split}_preds.npz no trae logits y el run está calibrado.")
    else:
        p = data["p"]
    table = pd.DataFrame({
        "path": data["path"], "y": data["y"].astype(int), "p": p,
        "camera": data["camera"], "weather": data["weather"],
        "date": data["date"], "time": data["time"], "slot_id": data["slot_id"],
    })
    return table


def decision(p: np.ndarray, t_L: float, t_H: float) -> np.ndarray:
    """'libre' si p < t_L, 'ocupado' si p >= t_H y 'por verificar' en medio."""
    p = np.asarray(p)
    return np.where(p >= t_H, "ocupado", np.where(p < t_L, "libre", "por verificar"))


def _pick(pool: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Una fila al azar (con la semilla dada) de un conjunto de candidatos."""
    if pool.empty:
        raise ValueError("No hay candidatos para esta regla.")
    return pool.loc[rng.choice(pool.index.to_numpy())]


def select_instances(table: pd.DataFrame, t_L: float, t_H: float, seed: int = SEED) -> pd.DataFrame:
    """Instancias locales con reglas fijas. Devuelve una fila por instancia, con su motivo.

    Reglas:
    1. acierto ocupado con p >= 0.999 (sol);
    2. acierto libre con p < 0.002 (sol);
    3. error ocupado→libre, el de menor p (el más seguro de su error);
    4. el caso más cercano al umbral t_H (en log-odds), de cualquier clase;
    5. acierto ocupado con lluvia, en una cámara distinta a la del error;
    6. error libre→ocupado, el de mayor p (el más seguro de su error).
    """
    rng = np.random.default_rng(seed)
    dec = decision(table.p.to_numpy(), t_L, t_H)
    df = table.assign(decision=dec)
    sunny = df.weather == "SUNNY"

    rows: list[pd.Series] = []

    def add(label: str, row: pd.Series) -> pd.Series:
        row = row.copy()
        row["motivo"] = label
        rows.append(row)
        return row

    add(HIT_BUSY, _pick(df[(df.y == 1) & (df.p >= HIGH_BUSY) & sunny], rng))
    add(HIT_FREE, _pick(df[(df.y == 0) & (df.p < HIGH_FREE) & sunny], rng))
    err = add(ERR_BUSY_AS_FREE, df[(df.y == 1) & (df.decision == "libre")].sort_values("p").iloc[0])

    logit = np.log(np.clip(df.p, 1e-12, 1 - 1e-12) / (1 - np.clip(df.p, 1e-12, 1 - 1e-12)))
    gap = (logit - np.log(t_H / (1 - t_H))).abs()
    add(NEAR_THRESHOLD, df.loc[gap.idxmin()])

    rain = df[(df.weather == "RAINY") & (df.y == 1) & (df.p >= HIGH_BUSY) & (df.camera != err.camera)]
    add(RAIN_HIT, _pick(rain, rng))
    add(ERR_FREE_AS_BUSY, df[(df.y == 0) & (df.decision == "ocupado")].sort_values("p").iloc[-1])

    out = pd.DataFrame(rows).reset_index().rename(columns={"index": "row"})
    return out[["motivo", "row", "path", "y", "p", "decision", "camera", "weather", "date", "time", "slot_id"]]


# ---------------------------------------------------------------------------
# Oclusión
# ---------------------------------------------------------------------------

def margin_score(logits: torch.Tensor, target: np.ndarray | list[int] | int) -> torch.Tensor:
    """Margen ``z_c - z_otra`` de la clase explicada (el mismo puntaje de Grad-CAM).

    ``target`` 1 explica "ocupado" (z_1 - z_0) y 0 explica "libre" (z_0 - z_1).
    """
    tgt = torch.as_tensor(np.asarray(target, dtype=np.int64), device=logits.device).reshape(-1)
    sign = (tgt * 2 - 1).to(logits.dtype)
    return sign * (logits[:, 1] - logits[:, 0])


@torch.no_grad()
def occlusion_map(model: nn.Module, x: torch.Tensor, target: int, patch: int = 56,
                  stride: int = 28, baseline: float = 0.0, batch_size: int = 64) -> np.ndarray:
    """Mapa de oclusión ``[H, W]``: caída promedio del margen al tapar cada ventana.

    - ``x`` es una imagen ya preprocesada ``[1, 3, H, W]``;
    - la ventana de ``patch`` píxeles se desliza con paso ``stride`` y se rellena con ``baseline``
      (0 es el color medio, porque la entrada ya viene normalizada);
    - a cada píxel se le asigna el promedio de la caída de las ventanas que lo cubren;
    - un valor alto significa que tapar esa zona le quita evidencia a la clase explicada.
    """
    _, _, height, width = x.shape
    starts_y = list(range(0, height - patch + 1, stride))
    starts_x = list(range(0, width - patch + 1, stride))
    base = margin_score(model(x), [target]).item()
    boxes = [(i, j) for i in starts_y for j in starts_x]
    drop = torch.zeros(height, width, device=x.device)
    count = torch.zeros(height, width, device=x.device)
    for k in range(0, len(boxes), batch_size):
        chunk = boxes[k:k + batch_size]
        batch = x.repeat(len(chunk), 1, 1, 1)
        for n, (i, j) in enumerate(chunk):
            batch[n, :, i:i + patch, j:j + patch] = baseline
        scores = margin_score(model(batch), [target] * len(chunk))
        for n, (i, j) in enumerate(chunk):
            drop[i:i + patch, j:j + patch] += base - scores[n]
            count[i:i + patch, j:j + patch] += 1
    return (drop / count.clamp_min(1)).cpu().numpy()


def normalize_map(m: np.ndarray) -> np.ndarray:
    """Lleva un mapa a [0, 1] (los valores negativos pasan a 0) para compararlo con Grad-CAM."""
    m = np.clip(m, 0, None)
    top = m.max()
    return m / top if top > 0 else m


# ---------------------------------------------------------------------------
# Fidelidad y concentración
# ---------------------------------------------------------------------------

def _square_mask(height: int, width: int, side: int, top: int, left: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[top:top + side, left:left + side] = True
    return mask


@torch.no_grad()
def deletion_fidelity(model: nn.Module, batch: torch.Tensor, cams: np.ndarray, target: np.ndarray,
                      fraction: float = 0.2, n_random: int = 5, baseline: float = 0.0,
                      seed: int = SEED) -> pd.DataFrame:
    """Qué tanto baja el margen al tapar la zona que marca el mapa, frente a otras zonas.

    Por imagen se tapa el ``fraction`` de los píxeles (rellenando con ``baseline``) en tres casos:
    - ``top``: los píxeles de mayor valor del mapa;
    - ``bottom``: los de menor valor;
    - ``azar``: un cuadrado de la misma área en una posición al azar (promedio de ``n_random``).
    Devuelve, por imagen, el margen original y la caída del margen en cada caso. Si la caída de
    ``top`` es mayor que la de ``azar`` y la de ``bottom``, el mapa señala evidencia real.
    """
    rng = np.random.default_rng(seed)
    device = batch.device
    n, _, height, width = batch.shape
    area = int(round(fraction * height * width))
    side = int(round(np.sqrt(area)))
    rows = []
    for i in range(n):
        x, cam, tgt = batch[i:i + 1], cams[i], int(target[i])
        order = np.argsort(cam.reshape(-1), kind="stable")
        top_mask = np.zeros(cam.size, dtype=bool)
        bottom_mask = np.zeros(cam.size, dtype=bool)
        top_mask[order[-area:]] = True
        bottom_mask[order[:area]] = True
        variants = [top_mask.reshape(height, width), bottom_mask.reshape(height, width)]
        for _ in range(n_random):
            top = int(rng.integers(0, height - side + 1))
            left = int(rng.integers(0, width - side + 1))
            variants.append(_square_mask(height, width, side, top, left))
        stack = x.repeat(len(variants), 1, 1, 1)
        for k, mask in enumerate(variants):
            stack[k][:, torch.from_numpy(mask).to(device)] = baseline
        scores = margin_score(model(torch.cat([x, stack])), [tgt] * (len(variants) + 1)).cpu().numpy()
        drops = float(scores[0]) - scores[1:]
        rows.append({"margen": float(scores[0]), "caida_top": float(drops[0]),
                     "caida_bottom": float(drops[1]), "caida_azar": float(drops[2:].mean())})
    return pd.DataFrame(rows)


def center_share(cams: np.ndarray, center_frac: float = 0.5) -> np.ndarray:
    """Parte de la atención (suma del mapa) que cae en el cuadrado central del parche.

    El cuadrado central tiene lado ``center_frac`` del lado del parche, o sea un área de
    ``center_frac ** 2`` del total. Si la atención fuera pareja, la parte sería igual a esa área.
    """
    _, height, width = cams.shape
    sh, sw = int(round(height * center_frac)), int(round(width * center_frac))
    top, left = (height - sh) // 2, (width - sw) // 2
    inside = cams[:, top:top + sh, left:left + sw].sum(axis=(1, 2))
    total = cams.sum(axis=(1, 2))
    return np.divide(inside, total, out=np.zeros_like(inside), where=total > 0)


# ---------------------------------------------------------------------------
# Baseline interpretable (HOG + regresión logística)
# ---------------------------------------------------------------------------

def hog_contribution_map(model: dict, path: str) -> np.ndarray:
    """Contribución de cada bloque HOG al log-odds de "ocupado" del modelo B1.

    La regresión logística es lineal: ``logit = b + sum_i w_i z_i``, con ``z`` el HOG
    estandarizado. Se suman los ``w_i z_i`` de los valores de cada bloque y queda un mapa
    ``[bloques, bloques]`` con signo (positivo empuja a ocupado, negativo a libre).
    """
    from cupo.classical import hog_one

    hog_cfg = model["hog"]
    feats = hog_one(path, **hog_cfg)
    scaler, clf = model["pipeline"].steps[0][1], model["pipeline"].steps[-1][1]
    contrib = clf.coef_[0] * ((feats - scaler.mean_) / scaler.scale_)
    cells = hog_cfg["image_size"] // hog_cfg["pixels_per_cell"]
    blocks = cells - hog_cfg["cells_per_block"] + 1
    return contrib.reshape(blocks, blocks, -1).sum(axis=2)


# ---------------------------------------------------------------------------
# Muestra estratificada para la explicación global
# ---------------------------------------------------------------------------

def stratified_sample(table: pd.DataFrame, per_stratum: int, seed: int = SEED) -> pd.DataFrame:
    """Muestra con el mismo número de imágenes por clase y clima (o todas si hay menos)."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in table.groupby(["y", "weather"], sort=True):
        k = min(per_stratum, len(group))
        parts.append(group.loc[np.sort(rng.choice(group.index.to_numpy(), size=k, replace=False))])
    return pd.concat(parts).reset_index().rename(columns={"index": "row"})
