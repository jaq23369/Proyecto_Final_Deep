"""Entrenamiento genérico de un clasificador libre/ocupado.

Uso:
    python -m cupo.train --config configs/X.yaml [--seed N] [--subset F] [--epochs N] [--run-name S]

Cada época mide en val la pérdida, accuracy, F1, AUC y el **valor de negocio** del escenario
``select.scenario`` con sus umbrales óptimos. ``best.pt`` se guarda en la época de máximo
valor en val (desempate: menor val loss) y el early stopping usa el mismo criterio.
Nunca lee test ni OOD. Al final escribe ``runs/<run_id>/`` (ver ``cupo.runs``).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from cupo import paths
from cupo.data import load_split, make_loader, subsample
from cupo.evaluate import (
    choose_thresholds, load_scenarios, logits_from_loader, probs_from_logits, write_split_outputs,
)
from cupo.losses import build_loss
from cupo.metrics import binary_metrics
from cupo.models import build_model, count_params, model_size_mb, set_train_mode
from cupo.runs import (
    choose_runs_root, create_run_dir, deep_merge, load_yaml, make_run_id, save_json, save_yaml,
)
from cupo.thresholds import optimize
from cupo.transforms import build_transforms
from cupo.utils import env_info, set_seed
from cupo.value_model import load_costs

DEFAULT_CONFIG: dict[str, Any] = {
    "name": "run",
    "notes": "",
    "model": {"name": "malexnet", "pretrained": False, "freeze": "none", "num_classes": 2},
    "data": {"input_size": 224, "augment": False, "batch_size": 128, "num_workers": 4, "subset": 1.0},
    "optim": {"name": "adamw", "lr": 1.0e-3, "weight_decay": 1.0e-4, "momentum": 0.9,
              "scheduler": "cosine", "warmup_epochs": 0, "min_lr_ratio": 0.0},
    "loss": {"label_smoothing": 0.0, "cost_weight_busy": 1.0},
    "train": {"epochs": 10, "early_stopping_patience": 3, "amp": True, "seed": 42,
              "deterministic": False, "cudnn_benchmark": False, "grad_clip": 0.0},
    "select": {"scenario": "comercial"},
    "eval": {"n_boot": 1000},
}

HISTORY_COLUMNS: list[str] = [
    "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro", "val_auc",
    "val_value", "lr", "epoch_time_s", "train_nll", "lr_start",
]
_VALUE_RTOL = 1e-9  # tolerancia para considerar dos valores iguales (desempate por val loss)
# Con pocos ejemplos (pruebas de humo) los workers tardan más en arrancar (spawn en Windows)
# que lo que ahorran; por debajo de este tamaño se carga en el proceso principal.
SMALL_DATASET = 5000
# train_acc de summary.json/history.csv: media de la época de entrenamiento con el modelo en
# modo train (aumento de datos y dropout activos), no una evaluación limpia sobre train.
TRAIN_ACC_NOTE = "media de la época con aumento/dropout"


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def resolve_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Config del YAML mezclada con los valores por defecto y los argumentos de la CLI."""
    cfg = deep_merge(DEFAULT_CONFIG, load_yaml(path))
    return deep_merge(cfg, overrides or {})


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    over: dict[str, Any] = {}
    if args.seed is not None:
        over.setdefault("train", {})["seed"] = args.seed
    if args.epochs is not None:
        over.setdefault("train", {})["epochs"] = args.epochs
    if args.subset is not None:
        over.setdefault("data", {})["subset"] = args.subset
    if args.n_boot is not None:
        over.setdefault("eval", {})["n_boot"] = args.n_boot
    if args.num_workers is not None:
        over.setdefault("data", {})["num_workers"] = args.num_workers
    return over


# ---------------------------------------------------------------------------
# Optimizador y scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, ocfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name = str(ocfg["name"]).lower()
    lr, wd = float(ocfg["lr"]), float(ocfg["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd,
                               momentum=float(ocfg.get("momentum", 0.9)), nesterov=True)
    raise ValueError(f"Optimizador desconocido: {name!r} (opciones: adamw, sgd)")


def lr_factor(step: int, total_steps: int, warmup_steps: int, schedule: str, min_ratio: float) -> float:
    """Factor del lr en el paso ``step``: warmup lineal y luego coseno (o constante)."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule in ("none", "constant", None):
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"Scheduler desconocido: {schedule!r} (opciones: cosine, none)")
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def build_scheduler(optimizer, ocfg: Mapping[str, Any], epochs: int, steps_per_epoch: int):
    """LambdaLR que se actualiza por iteración."""
    total = max(1, epochs * steps_per_epoch)
    warmup = int(round(float(ocfg.get("warmup_epochs", 0)) * steps_per_epoch))
    schedule = ocfg.get("scheduler", "cosine")
    min_ratio = float(ocfg.get("min_lr_ratio", 0.0))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total, warmup, schedule, min_ratio))


# ---------------------------------------------------------------------------
# Época de entrenamiento y validación
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device,
                    amp: bool, grad_clip: float, desc: str) -> dict[str, float]:
    """Una época de entrenamiento. Devuelve medias de la época (con aumento y dropout activos):

    - ``train_loss``: pérdida de optimización (con pesos de clase y label smoothing);
    - ``train_nll``: cross entropy sin pesos ni suavizado, comparable con ``val_loss``;
    - ``train_acc``: accuracy;
    - ``lr``: lr medio de la época; ``lr_start``: lr del primer paso.
    """
    set_train_mode(model)
    total_loss, total_nll, total_lr, correct, seen, steps = 0.0, 0.0, 0.0, 0, 0, 0
    lr_start = float(optimizer.param_groups[0]["lr"])
    bar = tqdm(loader, desc=desc, leave=False, mininterval=2.0, dynamic_ncols=True)
    for x, y in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        total_lr += float(optimizer.param_groups[0]["lr"])
        steps += 1
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        n = y.size(0)
        total_loss += float(loss.detach()) * n
        total_nll += float(nn.functional.cross_entropy(logits.detach().float(), y)) * n
        correct += int((logits.detach().argmax(1) == y).sum())
        seen += n
        bar.set_postfix(loss=f"{total_loss / seen:.4f}", refresh=False)
    seen = max(1, seen)
    return {"train_loss": total_loss / seen, "train_nll": total_nll / seen,
            "train_acc": correct / seen, "lr": total_lr / max(1, steps), "lr_start": lr_start}


def nll(logits: np.ndarray, y: np.ndarray) -> float:
    """Cross entropy media sin pesos ni suavizado (comparable entre configs)."""
    t = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    return float(nn.functional.cross_entropy(t, torch.from_numpy(y.astype(np.int64))))


def validate(model, loader, device, amp: bool, scenario_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Pérdida, métricas y valor óptimo en val para la época actual."""
    logits, y = logits_from_loader(model, loader, device, amp)
    p = probs_from_logits(logits)
    bm = binary_metrics(y, p, 0.5)
    best = optimize(y, p, scenario_cfg)
    return {
        "logits": logits, "y": y, "p": p,
        "val_loss": nll(logits, y), "val_acc": bm["accuracy"], "val_f1_macro": bm["f1_macro"],
        "val_auc": bm["auc"], "val_value": float(best["value"]),
    }


def is_better(value: float, loss: float, best_value: float, best_loss: float) -> bool:
    """Mayor valor en val; si empata (tolerancia relativa), menor val loss."""
    if not np.isfinite(best_value):
        return True
    tol = _VALUE_RTOL * max(1.0, abs(best_value))
    if value > best_value + tol:
        return True
    return abs(value - best_value) <= tol and loss < best_loss


# ---------------------------------------------------------------------------
# Artefactos
# ---------------------------------------------------------------------------

def plot_curves(history: pd.DataFrame, out_path: Path, scenario: str) -> None:
    """``curves.png``: pérdida, accuracy y valor en val por época."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep = history["epoch"]
    # train_nll y val_loss son la misma CE sin pesos ni suavizado (comparables);
    # train_loss es la pérdida de optimización (con pesos de clase / label smoothing).
    axes[0].plot(ep, history["train_nll"], "o-", label="train NLL (media de la época)")
    axes[0].plot(ep, history["val_loss"], "o-", label="val NLL")
    axes[0].plot(ep, history["train_loss"], "--", color="gray", alpha=0.7,
                 label="train_loss (pérdida de optimización)")
    axes[0].set_title("Pérdida (NLL sin pesos)")
    axes[1].plot(ep, history["train_acc"], "o-", label="train (media de la época, con aumento)")
    axes[1].plot(ep, history["val_acc"], "o-", label="val")
    axes[1].set_title("Accuracy")
    axes[2].plot(ep, history["val_value"], "o-", color="tab:green", label="val")
    axes[2].set_title(f"Valor en val ({scenario})")
    from matplotlib.ticker import MaxNLocator

    for ax in axes:
        ax.set_xlabel("época")
        if len(ep) <= 20:
            ax.set_xticks(list(ep))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def train_cost(train_time_s: float, rate_key: str = "gpu_usd_per_hour") -> tuple[float, float]:
    """(costo en USD, costo en Q) del tiempo de entrenamiento según ``configs/costs.yaml``."""
    costs = load_costs()
    usd = train_time_s / 3600.0 * float(costs[rate_key])
    return usd, usd * float(costs["gtq_per_usd"])


def build_summary(run_dir: Path, cfg: Mapping[str, Any], params: tuple[int, int], size_mb: float,
                  best_epoch: int | None, epochs_run: int, train_time_s: float, gpu: str | None,
                  cost: tuple[float, float], metrics_val: Mapping[str, Any],
                  values: Mapping[str, Mapping[str, Any]], train_acc: float | None) -> dict[str, Any]:
    """Contenido de ``summary.json`` (columnas de la tabla de iteraciones)."""
    binary = metrics_val["binary"]
    uvg = values.get("uvg", {})
    return {
        "run_id": run_dir.name,
        "name": cfg["name"],
        "notes": cfg.get("notes", ""),
        "model": cfg["model"]["name"],
        "freeze": cfg["model"].get("freeze"),
        "params_total": params[0],
        "params_trainable": params[1],
        "size_mb": size_mb,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "train_time_s": train_time_s,
        "gpu": gpu,
        "train_cost_usd": cost[0],
        "train_cost_q": cost[1],
        "seed": cfg["train"]["seed"],
        "subset": cfg["data"]["subset"],
        "select_scenario": cfg["select"]["scenario"],
        "train_acc": train_acc,
        "train_acc_note": TRAIN_ACC_NOTE,
        "val_accuracy": binary["accuracy"],
        "val_f1_macro": binary["f1_macro"],
        "val_auc": binary["auc"],
        "val_rate_busy_as_free": binary["rate_busy_as_free"],
        "val_value_comercial": values.get("comercial", {}).get("objective_value"),
        "val_value_comercial_ci95": values.get("comercial", {}).get("ci95", {}).get("objective"),
        "val_value_uvg": uvg.get("objective_value"),
        "val_value_uvg_ci95": uvg.get("ci95", {}).get("objective"),
        "val_hours_uvg": uvg.get("components", {}).get("horas_ahorradas_semestre"),
    }


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------

def load_data(cfg: Mapping[str, Any], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train y val (con submuestra estratificada si ``data.subset`` < 1). Nunca test ni OOD."""
    train_df, val_df = load_split("train"), load_split("val")
    frac = float(cfg["data"]["subset"])
    if frac < 1:
        train_df, val_df = subsample(train_df, frac, seed), subsample(val_df, frac, seed)
    return train_df, val_df


def effective_workers(n_items: int, requested: int) -> int:
    """Workers del DataLoader: 0 para conjuntos pequeños, ``requested`` en otro caso."""
    return 0 if n_items <= SMALL_DATASET else requested


def run_training(cfg: dict[str, Any], run_name: str | None = None, runs_root: Path | None = None) -> Path:
    """Entrena según ``cfg`` y escribe el run completo; devuelve la carpeta del run."""
    seed = int(cfg["train"]["seed"])
    set_seed(seed, deterministic=bool(cfg["train"]["deterministic"]))
    # cudnn.benchmark tarda ~30 s por cada forma de batch nueva (MobileNet/EfficientNet en la
    # RTX 2060) y casi no acelera; por eso está apagado por defecto.
    torch.backends.cudnn.benchmark = bool(cfg["train"]["cudnn_benchmark"]) and not cfg["train"]["deterministic"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"

    run_dir = create_run_dir(run_name or make_run_id(cfg["name"], seed), runs_root)
    save_yaml(run_dir / "config.yaml", cfg)
    save_json(run_dir / "env.json", env_info())
    print(f"Run: {run_dir}")

    train_df, val_df = load_data(cfg, seed)
    dcfg = cfg["data"]
    size, workers, bs = int(dcfg["input_size"]), int(dcfg["num_workers"]), int(dcfg["batch_size"])
    train_loader = make_loader(train_df, build_transforms(size, train=True, augment=bool(dcfg["augment"])),
                               batch_size=bs, shuffle=True, seed=seed,
                               num_workers=effective_workers(len(train_df), workers))
    val_loader = make_loader(val_df, build_transforms(size, train=False, augment=False),
                             batch_size=bs * 2, shuffle=False, seed=seed,
                             num_workers=effective_workers(len(val_df), workers))
    print(f"train={len(train_df):,}  val={len(val_df):,}  device={device}  amp={amp}  "
          f"workers={train_loader.num_workers}/{val_loader.num_workers}")

    mcfg = cfg["model"]
    model = build_model(mcfg["name"], num_classes=int(mcfg["num_classes"]),
                        pretrained=bool(mcfg["pretrained"]), freeze=mcfg["freeze"], input_size=size)
    model.to(device)
    params = count_params(model)
    print(f"Modelo {mcfg['name']} (freeze={mcfg['freeze']}): {params[0]:,} parámetros, "
          f"{params[1]:,} entrenables")

    criterion = build_loss(cfg["loss"]).to(device)
    optimizer = build_optimizer(model, cfg["optim"])
    epochs = int(cfg["train"]["epochs"])
    scheduler = build_scheduler(optimizer, cfg["optim"], epochs, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    scenarios = load_scenarios(cfg["select"]["scenario"])
    primary = cfg["select"]["scenario"]
    patience = int(cfg["train"]["early_stopping_patience"])

    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"value": -np.inf, "loss": np.inf, "epoch": None, "val": None, "train_acc": None}
    stale = 0
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, amp,
            float(cfg["train"]["grad_clip"]), desc=f"época {epoch}/{epochs}")
        val = validate(model, val_loader, device, amp, scenarios[primary])
        tr_loss, tr_acc = tr["train_loss"], tr["train_acc"]
        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
               **{k: val[k] for k in ("val_loss", "val_acc", "val_f1_macro", "val_auc", "val_value")},
               "lr": tr["lr"], "epoch_time_s": time.perf_counter() - t0,
               "train_nll": tr["train_nll"], "lr_start": tr["lr_start"]}
        history.append(row)
        improved = is_better(val["val_value"], val["val_loss"], best["value"], best["loss"])
        if improved:
            best.update(value=val["val_value"], loss=val["val_loss"], epoch=epoch, val=val, train_acc=tr_acc)
            torch.save({"state_dict": model.state_dict(), "model_cfg": dict(mcfg),
                        "input_size": size, "epoch": epoch}, run_dir / "best.pt")
            stale = 0
        else:
            stale += 1
        print(f"época {epoch:>2}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val['val_loss']:.4f} "
              f"acc={val['val_acc']:.4f} f1={val['val_f1_macro']:.4f} valor={val['val_value']:,.1f}"
              f"{' *' if improved else ''}  ({row['epoch_time_s']:.1f} s)")
        pd.DataFrame(history, columns=HISTORY_COLUMNS).to_csv(run_dir / "history.csv", index=False)
        if patience > 0 and stale >= patience:
            print(f"Early stopping: {patience} épocas sin mejorar el valor en val")
            break
    train_time_s = time.perf_counter() - t_start

    hist_df = pd.DataFrame(history, columns=HISTORY_COLUMNS)
    plot_curves(hist_df, run_dir / "curves.png", primary)

    # Artefactos de val con las predicciones de la mejor época (la misma de best.pt)
    bval = best["val"]
    thresholds = choose_thresholds(bval["y"], bval["p"], scenarios)
    save_json(run_dir / "thresholds.json", thresholds)
    n_boot = int(cfg["eval"]["n_boot"])
    metrics_val, values = write_split_outputs(
        run_dir, "val", val_df, bval["logits"], bval["p"], thresholds, scenarios, n_boot,
        extra={"temperature": 1.0, "calibrated": False, "model_kind": "torch", "epoch": best["epoch"]})

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    summary = build_summary(run_dir, cfg, params, model_size_mb(model), best["epoch"], len(history),
                            train_time_s, gpu, train_cost(train_time_s), metrics_val, values,
                            best["train_acc"])
    save_json(run_dir / "summary.json", summary)
    print(f"Mejor época {best['epoch']}: valor {primary} = {best['value']:,.1f}; "
          f"tiempo {train_time_s:.1f} s; costo USD {summary['train_cost_usd']:.4f}")
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un clasificador de espacios de parqueo.")
    parser.add_argument("--config", required=True, help="configs/<nombre>.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subset", type=float, default=None, help="fracción de train y val (humo)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", default=None, help="nombre de la carpeta del run (reemplaza el run_id)")
    parser.add_argument("--runs-dir", default=None,
                        help=f"carpeta de runs (por defecto {paths.runs_dir()}; con --subset < 1, runs/_smoke)")
    parser.add_argument("--n-boot", type=int, default=None, help="réplicas bootstrap de los IC95")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = resolve_config(args.config, cli_overrides(args))
    runs_root = choose_runs_root(args.runs_dir, cfg["data"]["subset"])
    if args.runs_dir is None and float(cfg["data"]["subset"]) < 1:
        print(f"subset={cfg['data']['subset']} < 1: prueba de humo, el run se guarda en {runs_root}")
    run_training(cfg, run_name=args.run_name, runs_root=runs_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
