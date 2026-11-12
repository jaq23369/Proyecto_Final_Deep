"""Grad-CAM sobre aciertos y errores de validación de un run de PyTorch.

Uso:
    python scripts/gradcam.py --run runs/<id> --split val --n 16 [--out results/figures]

- Implementación propia con hooks sobre la última capa convolucional:
  mAlexNet → conv3 + ReLU (post-activación); MobileNetV3 y EfficientNet → último bloque
  de ``features``; ResNet → último bloque de ``layer4``.
- Puntaje explicado: margen ``z_c - z_otra`` (el log-odds que decide p) de la clase de la
  decisión del run: ocupado si p >= t_H, libre si p < t_L; en "por verificar", la clase
  más probable (p >= 0.5 → ocupado).
- Ejemplos de ``val_preds.npz`` con la temperatura y los umbrales del run (escenario de
  selección): la mitad errores, primero ocupado→libre (y=1 con p < t_L, los más seguros
  primero) y luego libre→ocupado (y=0 con p >= t_H); hasta 2 "por verificar"
  (t_L <= p < t_H) si existen; el resto, aciertos ocupados y libres elegidos al azar con
  semilla fija.
- Si el run está calibrado, ``val_preds.npz`` debe traer ``logits`` (p se recalcula con T).
- Salida: ``gradcam_<run_id>_val.png`` (imagen | mapa superpuesto, con p, y, clima y
  categoría) y ``gradcam_<run_id>_val.csv`` con los ejemplos elegidos.
- Solo ``val``: test y CNRPark A/B (ood) se rechazan antes del congelamiento.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402

from cupo import paths  # noqa: E402
from cupo.evaluate import probs_from_logits  # noqa: E402
from cupo.runs import LoadedRun, load_npz, load_run  # noqa: E402
from cupo.transforms import preprocess_numpy  # noqa: E402

SEED = 0
CLASS_NAMES = ("libre", "ocupado")
CAT_BUSY_AS_FREE = "error ocupado->libre"
CAT_FREE_AS_BUSY = "error libre->ocupado"
CAT_HIT_BUSY = "acierto ocupado"
CAT_HIT_FREE = "acierto libre"
CAT_VERIFY = "por verificar"
N_VERIFY = 2      # ejemplos "por verificar" (si existen)
COLS_PER_ROW = 4  # ejemplos por fila (cada uno ocupa 2 paneles)


# ---------------------------------------------------------------------------
# Capa objetivo y Grad-CAM
# ---------------------------------------------------------------------------

def target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Última capa convolucional (o bloque) según la arquitectura."""
    if arch == "malexnet":
        return model.features[2][1]          # ReLU de conv3 (post-activación)
    if arch.startswith("resnet"):
        return model.layer4[-1]
    if arch.startswith(("mobilenet_v3", "efficientnet")):
        return model.features[-1]            # última Conv2dNormActivation
    raise ValueError(f"Arquitectura sin capa objetivo definida: {arch!r}")


def disable_inplace(model: nn.Module) -> None:
    """Desactiva las activaciones in-place para que la activación guardada no cambie."""
    for module in model.modules():
        if isinstance(getattr(module, "inplace", None), bool):
            module.inplace = False


class GradCAM:
    """Grad-CAM: pesos = media espacial del gradiente; mapa = ReLU(sum_k w_k · A_k)."""

    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model = model
        self._act: torch.Tensor | None = None
        self._handle = layer.register_forward_hook(self._save)

    def _save(self, _module: nn.Module, _inp, out: torch.Tensor) -> None:
        self._act = out

    def close(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mapas ``[N, H, W]`` en [0, 1] (clase ``target`` por ejemplo) y logits ``[N, 2]``."""
        with torch.enable_grad():
            logits = self.model(x)
            tgt = torch.tensor(np.asarray(target, dtype=np.int64), device=logits.device)
            sign = (tgt * 2 - 1).to(logits.dtype)        # +1 explica ocupado, -1 explica libre
            score = (sign * (logits[:, 1] - logits[:, 0])).sum()
            grads = torch.autograd.grad(score, self._act)[0]
        act = self._act.detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
        cam = ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(cam)
        return cam.cpu().numpy(), logits.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# Selección de ejemplos
# ---------------------------------------------------------------------------

def operating_thresholds(run: LoadedRun) -> tuple[str, float, float]:
    scenario = run.config.get("select", {}).get("scenario", "comercial")
    if scenario not in run.thresholds:
        scenario = next(iter(run.thresholds))
    thr = run.thresholds[scenario]
    return scenario, float(thr["t_L"]), float(thr["t_H"])


def load_val_table(run: LoadedRun) -> pd.DataFrame:
    """``val_preds.npz`` con ``p`` recalculada con la temperatura del run."""
    data = load_npz(run.run_dir / "val_preds.npz")
    if "logits" in data:
        p = probs_from_logits(data["logits"], run.temperature)
    elif run.calibrated:
        raise ValueError(f"El run {run.run_id} está calibrado pero val_preds.npz no tiene logits: "
                         "no se puede aplicar la temperatura.")
    else:
        p = data["p"]
    return pd.DataFrame({"path": data["path"], "y": data["y"].astype(int), "p": p,
                         "weather": data["weather"], "camera": data["camera"]})


def categorize(df: pd.DataFrame, t_L: float, t_H: float) -> pd.Series:
    cat = pd.Series(CAT_VERIFY, index=df.index, dtype=object)
    cat[(df.y == 1) & (df.p < t_L)] = CAT_BUSY_AS_FREE
    cat[(df.y == 0) & (df.p >= t_H)] = CAT_FREE_AS_BUSY
    cat[(df.y == 1) & (df.p >= t_H)] = CAT_HIT_BUSY
    cat[(df.y == 0) & (df.p < t_L)] = CAT_HIT_FREE
    return cat


def decision_class(p: np.ndarray, t_L: float, t_H: float) -> np.ndarray:
    """Clase de la decisión del run: 1 si p >= t_H, 0 si p < t_L; en verificar, p >= 0.5."""
    p = np.asarray(p)
    return np.where(p >= t_H, 1, np.where(p < t_L, 0, (p >= 0.5).astype(int))).astype(int)


def _random_rows(pool: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    idx = rng.choice(pool.index.to_numpy(), size=min(k, len(pool)), replace=False)
    return pool.loc[np.sort(idx)]


def select_examples(df: pd.DataFrame, n: int, t_L: float, t_H: float, seed: int = SEED) -> pd.DataFrame:
    """Mitad errores (ocupado→libre primero), hasta 2 por verificar y el resto aciertos."""
    df = df.assign(category=categorize(df, t_L, t_H))
    rng = np.random.default_rng(seed)
    busy_free = df[df.category == CAT_BUSY_AS_FREE].sort_values("p")
    free_busy = df[df.category == CAT_FREE_AS_BUSY].sort_values("p", ascending=False)
    errors = pd.concat([busy_free, free_busy]).head(int(np.ceil(n / 2)))
    verify = _random_rows(df[df.category == CAT_VERIFY], min(N_VERIFY, n - len(errors)), rng)
    n_hits = n - len(errors) - len(verify)
    picks = [verify]
    for k, cat in enumerate((CAT_HIT_BUSY, CAT_HIT_FREE)):
        pool = df[df.category == cat]
        want = n_hits // 2 + (n_hits % 2 if k == 0 else 0)
        picks.append(_random_rows(pool, want, rng))
    sel = pd.concat([errors, *picks]).head(n)
    sel = _fill_up(df, sel, n, rng)
    return sel.reset_index(drop=True)


def _fill_up(df: pd.DataFrame, sel: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Si alguna categoría queda vacía o corta, completa hasta ``n`` con las disponibles y avisa."""
    empty = [c for c in (CAT_BUSY_AS_FREE, CAT_FREE_AS_BUSY, CAT_VERIFY, CAT_HIT_BUSY, CAT_HIT_FREE)
             if not (df.category == c).any()]
    if empty:
        print(f"Aviso: categorías sin ejemplos en val con estos umbrales: {', '.join(empty)}")
    missing = n - len(sel)
    if missing <= 0:
        return sel
    rest = df.drop(index=sel.index)
    extra = _random_rows(rest, missing, rng)
    print(f"Aviso: se completaron {len(extra)} ejemplos con las categorías disponibles "
          f"({len(sel) + len(extra)} de {n}).")
    return pd.concat([sel, extra])


# ---------------------------------------------------------------------------
# Imágenes y figura
# ---------------------------------------------------------------------------

def resolve_patch_path(path: str) -> Path:
    """Ruta del patch; si el npz se generó en otra máquina, se reubica bajo ``DATA_ROOT``."""
    p = Path(path)
    if p.exists():
        return p
    parts = Path(path.replace("\\", "/")).parts
    if paths.EXT_DIRNAME in parts:
        return paths.data_root().joinpath(*parts[parts.index(paths.EXT_DIRNAME):])
    return p


def data_relative(path: str) -> str:
    """Ruta relativa a ``DATA_ROOT`` (``CNR-EXT/PATCHES/...``) para no guardar rutas locales."""
    full = resolve_patch_path(path)
    try:
        return full.resolve().relative_to(paths.data_root()).as_posix()
    except ValueError:
        parts = Path(path.replace("\\", "/")).parts
        if paths.EXT_DIRNAME in parts:
            return "/".join(parts[parts.index(paths.EXT_DIRNAME):])
        return full.name


def load_images(paths_: Sequence[str], input_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Batch preprocesado (como en evaluación) y las imágenes RGB redimensionadas para mostrar."""
    batch, shown = [], []
    for path in paths_:
        with Image.open(resolve_patch_path(path)) as im:
            batch.append(preprocess_numpy(im, input_size))
            shown.append(np.asarray(im.convert("RGB").resize((input_size, input_size),
                                                             Image.Resampling.BILINEAR)))
    return np.concatenate(batch), shown


def plot_grid(sel: pd.DataFrame, images: list[np.ndarray], cams: np.ndarray, title: str, out: Path) -> None:
    n = len(sel)
    rows = int(np.ceil(n / COLS_PER_ROW))
    fig, axes = plt.subplots(rows, 2 * COLS_PER_ROW, figsize=(2.0 * 2 * COLS_PER_ROW, 2.45 * rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, row in sel.iterrows():
        r, c = divmod(i, COLS_PER_ROW)
        ax_img, ax_cam = axes[r, 2 * c], axes[r, 2 * c + 1]
        color = ("firebrick" if row.category.startswith("error")
                 else "darkorange" if row.category == CAT_VERIFY else "darkgreen")
        ax_img.imshow(images[i])
        ax_img.set_title(f"{row.category}\n{row.weather} · cám {row.camera}", fontsize=7, color=color)
        ax_cam.imshow(images[i])
        ax_cam.imshow(cams[i], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax_cam.set_title(f"p={row.p:.3f}  y={CLASS_NAMES[int(row.y)]}\nCAM: {CLASS_NAMES[int(row.explained)]}",
                         fontsize=7, color=color)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM sobre aciertos y errores de val.")
    parser.add_argument("--run", required=True, help="runs/<run_id> o solo el run_id")
    parser.add_argument("--split", default="val", choices=["val", "test", "ood"])
    parser.add_argument("--n", type=int, default=16, help="número de ejemplos")
    parser.add_argument("--out", default=str(paths.results_dir() / "figures"), help="carpeta de salida")
    parser.add_argument("--device", default=None, help="cuda o cpu (por defecto cuda si hay)")
    args = parser.parse_args(argv)
    if args.split != "val":
        parser.error(f"--split {args.split} está bloqueado: Grad-CAM solo se calcula en val "
                     "(test y CNRPark A/B no se usan antes del congelamiento).")
    if args.n < 1:
        parser.error("--n debe ser >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run = load_run(args.run, device=device)
    if run.kind != "torch":
        print(f"Error: el run {run.run_id} es clásico (model.joblib); Grad-CAM requiere una CNN.",
              file=sys.stderr)
        return 1
    arch = run.config.get("model", {}).get("name", "")
    scenario, t_L, t_H = operating_thresholds(run)
    try:
        table = load_val_table(run)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    sel = select_examples(table, args.n, t_L, t_H)
    if sel.empty:
        print("Error: no hay ejemplos para mostrar en val_preds.npz.", file=sys.stderr)
        return 1

    model = run.model.float().eval()
    disable_inplace(model)
    cam_fn = GradCAM(model, target_layer(model, arch))
    batch, images = load_images(sel["path"].tolist(), run.input_size)
    sel["explained"] = decision_class(sel["p"].to_numpy(), t_L, t_H)
    cams, logits = cam_fn(torch.from_numpy(batch).to(device), sel["explained"].to_numpy())
    cam_fn.close()
    sel["p_recomputed"] = probs_from_logits(logits, run.temperature)

    out_dir = Path(args.out)
    stem = f"gradcam_{run.run_id}_val"
    title = (f"Grad-CAM {run.config.get('name', run.run_id)} ({arch}) en val | escenario {scenario}: "
             f"t_L={t_L:.2f}, t_H={t_H:.2f}, T={run.temperature:.3f}")
    plot_grid(sel, images, cams, title, out_dir / f"{stem}.png")
    sel.assign(path=sel["path"].map(data_relative)).to_csv(out_dir / f"{stem}.csv", index=False)

    counts = sel["category"].value_counts().to_dict()
    gap = float(np.abs(sel["p_recomputed"] - sel["p"]).max())
    print(f"{len(sel)} ejemplos: {counts}")
    print(f"max |p(npz) - p(recalculada fp32)| = {gap:.2e}")
    print(f"Figura: {out_dir / (stem + '.png')}")
    print(f"Tabla:  {out_dir / (stem + '.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
