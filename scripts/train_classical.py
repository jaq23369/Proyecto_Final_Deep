"""B1: baseline clásico sin deep learning (HOG + StandardScaler + regresión logística).

Uso:
    python scripts/train_classical.py --config configs/b1_hog_lr.yaml [--subset F] [--seed N]

Cada imagen se pasa a gris y se redimensiona a 64x64; se extrae HOG (9 orientaciones,
celdas de 8x8, bloques de 2x2) en paralelo con joblib. C se elige en val dentro de
``model.C_grid`` por el valor de negocio del escenario ``select.scenario`` (desempate:
menor log loss). Escribe la misma estructura de ``runs/<run_id>/`` que ``cupo.train``,
con ``model.joblib`` en lugar de ``best.pt``. Nunca lee test ni OOD.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cupo import paths
from cupo.evaluate import (
    choose_thresholds, classical_logits, load_scenarios, probs_from_logits, write_split_outputs,
)
from cupo.metrics import binary_metrics
from cupo.classical import HOG_DEFAULTS, hog_features
from cupo.runs import (
    choose_runs_root, create_run_dir, deep_merge, load_yaml, make_run_id, save_json, save_yaml,
)
from cupo.thresholds import optimize
from cupo.train import build_summary, load_data, train_cost
from cupo.utils import env_info, set_seed

DEFAULT_CONFIG: dict[str, Any] = {
    "name": "b1_hog_lr",
    "notes": "",
    "model": {"name": "hog_lr", "C_grid": [0.01, 0.1, 1.0], "solver": "lbfgs", "max_iter": 2000,
              "hog": dict(HOG_DEFAULTS)},
    "data": {"subset": 1.0, "n_jobs": -1},
    "train": {"seed": 42},
    "select": {"scenario": "comercial"},
    "eval": {"n_boot": 1000},
}


def extract(df: pd.DataFrame, hog_cfg: Mapping[str, Any], n_jobs: int, label: str) -> np.ndarray:
    """HOG en paralelo mostrando el tiempo."""
    t0 = time.perf_counter()
    feats = hog_features(df["path"].tolist(), dict(hog_cfg), n_jobs=n_jobs)
    dt = time.perf_counter() - t0
    print(f"HOG {label}: {feats.shape[0]:,} imágenes x {feats.shape[1]} features en {dt:.1f} s "
          f"({feats.shape[0] / max(dt, 1e-9):,.0f} img/s)")
    return feats


def fit_candidate(C: float, X: np.ndarray, y: np.ndarray, mcfg: Mapping[str, Any], seed: int) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, solver=mcfg["solver"], max_iter=int(mcfg["max_iter"]),
                                  random_state=seed)),
    ])
    return pipe.fit(X, y)


def search_C(X_tr, y_tr, X_val, y_val, mcfg, seed, scenario_cfg) -> tuple[list[dict], dict]:
    """Ajusta cada C y lo evalúa en val; devuelve la tabla y el mejor candidato."""
    rows, best = [], None
    for C in mcfg["C_grid"]:
        t0 = time.perf_counter()
        pipe = fit_candidate(float(C), X_tr, y_tr, mcfg, seed)
        fit_s = time.perf_counter() - t0
        bundle = {"pipeline": pipe}
        logits = classical_logits(bundle, X_val)
        p = probs_from_logits(logits)
        bm = binary_metrics(y_val, p, 0.5)
        value = float(optimize(y_val, p, scenario_cfg)["value"])
        p_tr = probs_from_logits(classical_logits(bundle, X_tr))
        train_acc = float(np.mean((p_tr >= 0.5) == (y_tr == 1)))
        train_nll = float(log_loss(y_tr, np.clip(p_tr, 1e-7, 1 - 1e-7), labels=[0, 1]))
        row = {"C": float(C), "train_acc": train_acc, "train_nll": train_nll, "val_loss": bm["log_loss"],
               "val_acc": bm["accuracy"], "val_f1_macro": bm["f1_macro"], "val_auc": bm["auc"],
               "val_value": value, "fit_time_s": fit_s}
        rows.append(row)
        print(f"C={C:<6} train_acc={train_acc:.4f} val_acc={bm['accuracy']:.4f} "
              f"f1={bm['f1_macro']:.4f} valor={value:,.1f} ({fit_s:.1f} s)")
        better = best is None or value > best["row"]["val_value"] or (
            value == best["row"]["val_value"] and row["val_loss"] < best["row"]["val_loss"])
        if better:
            best = {"row": row, "pipeline": pipe, "logits": logits, "p": p}
    return rows, best


def plot_search(table: pd.DataFrame, out_path: Path, scenario: str) -> None:
    """``curves.png`` del baseline: métricas y valor en val frente a C."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogx(table["C"], table["train_nll"], "o-", label="train")
    axes[0].semilogx(table["C"], table["val_loss"], "o-", label="val")
    axes[0].set_title("Log loss")
    axes[1].semilogx(table["C"], table["train_acc"], "o-", label="train")
    axes[1].semilogx(table["C"], table["val_acc"], "o-", label="val")
    axes[1].set_title("Accuracy")
    axes[2].semilogx(table["C"], table["val_value"], "o-", color="tab:green", label="val")
    axes[2].set_title(f"Valor en val ({scenario})")
    for ax in axes:
        ax.set_xlabel("C")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def history_frame(best_row: Mapping[str, Any], train_time_s: float) -> pd.DataFrame:
    """``history.csv`` con el formato de los runs de torch (una fila: el modelo elegido)."""
    return pd.DataFrame([{
        "epoch": 1, "train_loss": np.nan, "train_acc": best_row["train_acc"],
        "val_loss": best_row["val_loss"], "val_acc": best_row["val_acc"],
        "val_f1_macro": best_row["val_f1_macro"], "val_auc": best_row["val_auc"],
        "val_value": best_row["val_value"], "lr": np.nan, "epoch_time_s": train_time_s,
        "train_nll": best_row["train_nll"], "lr_start": np.nan,
    }])


def run(cfg: dict[str, Any], run_name: str | None = None, runs_root: Path | None = None) -> Path:
    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    run_dir = create_run_dir(run_name or make_run_id(cfg["name"], seed), runs_root)
    save_yaml(run_dir / "config.yaml", cfg)
    save_json(run_dir / "env.json", env_info())
    print(f"Run: {run_dir}")

    train_df, val_df = load_data(cfg, seed)
    print(f"train={len(train_df):,}  val={len(val_df):,}")
    mcfg, n_jobs = cfg["model"], int(cfg["data"]["n_jobs"])
    primary = cfg["select"]["scenario"]
    scenarios = load_scenarios(primary)

    t_start = time.perf_counter()
    X_tr = extract(train_df, mcfg["hog"], n_jobs, "train")
    X_val = extract(val_df, mcfg["hog"], n_jobs, "val")
    y_tr, y_val = train_df["label"].to_numpy(int), val_df["label"].to_numpy(int)
    rows, best = search_C(X_tr, y_tr, X_val, y_val, mcfg, seed, scenarios[primary])
    train_time_s = time.perf_counter() - t_start

    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "c_search.csv", index=False)
    history_frame(best["row"], train_time_s).to_csv(run_dir / "history.csv", index=False)
    plot_search(table, run_dir / "curves.png", primary)

    bundle = {"pipeline": best["pipeline"], "hog": dict(mcfg["hog"]), "C": best["row"]["C"]}
    joblib.dump(bundle, run_dir / "model.joblib")
    print(f"C elegido en val: {best['row']['C']}")

    thresholds = choose_thresholds(y_val, best["p"], scenarios)
    save_json(run_dir / "thresholds.json", thresholds)
    metrics_val, values = write_split_outputs(
        run_dir, "val", val_df, best["logits"], best["p"], thresholds, scenarios,
        int(cfg["eval"]["n_boot"]),
        extra={"temperature": 1.0, "calibrated": False, "model_kind": "classical", "C": best["row"]["C"]})

    lr = best["pipeline"].named_steps["lr"]
    n_params = int(lr.coef_.size + lr.intercept_.size)
    size_mb = (run_dir / "model.joblib").stat().st_size / 1024**2
    summary = build_summary(run_dir, cfg, (n_params, n_params), size_mb, None, 1, train_time_s, None,
                            train_cost(train_time_s, "cpu_server_usd_per_hour"), metrics_val, values,
                            best["row"]["train_acc"])
    summary["C"] = best["row"]["C"]
    summary["train_acc_note"] = "accuracy del modelo final sobre train (sin aumento)"
    save_json(run_dir / "summary.json", summary)
    print(f"Valor {primary} en val = {values[primary]['objective_value']:,.1f}; tiempo {train_time_s:.1f} s")
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline B1: HOG + regresión logística.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--subset", type=float, default=None, help="fracción de train y val (humo)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None, help="nombre de la carpeta del run (reemplaza el run_id)")
    parser.add_argument("--runs-dir", default=None,
                        help=f"carpeta de runs (por defecto {paths.runs_dir()}; con --subset < 1, runs/_smoke)")
    parser.add_argument("--n-boot", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = deep_merge(DEFAULT_CONFIG, load_yaml(args.config))
    over: dict[str, Any] = {}
    if args.subset is not None:
        over["data"] = {"subset": args.subset}
    if args.seed is not None:
        over["train"] = {"seed": args.seed}
    if args.n_boot is not None:
        over["eval"] = {"n_boot": args.n_boot}
    cfg = deep_merge(cfg, over)
    runs_root = choose_runs_root(args.runs_dir, cfg["data"]["subset"])
    if args.runs_dir is None and float(cfg["data"]["subset"]) < 1:
        print(f"subset={cfg['data']['subset']} < 1: prueba de humo, el run se guarda en {runs_root}")
    run(cfg, run_name=args.run_name, runs_root=runs_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
