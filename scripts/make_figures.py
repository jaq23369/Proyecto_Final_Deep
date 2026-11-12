"""Figuras de resultados de las iteraciones → ``results/figures/``.

Uso:
    python scripts/make_figures.py [--runs-dir runs] [--include-smoke] [--out results/figures]
                                   [--with-table [--table-out results]]

Con ``--with-table`` también genera la tabla de iteraciones (``results_table.py``): un solo
comando deja la tabla y las figuras.

Usa la misma lectura de runs que ``scripts/results_table.py`` (summary.json y, si existe,
calibration.json; probabilidades de ``val_preds.npz`` con la temperatura del run). El mejor
run es el de mayor valor comercial en val entre los runs con el conjunto completo
(``subset`` = 1); los subconjuntos (pruebas de humo) solo se usan si no hay runs completos y
entonces las figuras lo advierten. Todo se calcula en val; las figuras de test y OOD solo se
generan si el run ya tiene ``test_preds.npz`` / ``ood_preds.npz`` (evaluación única tras el
congelamiento). Si una figura falla se avisa y se sigue con las demás.

Figuras:
- ``curvas_entrenamiento.png``: train NLL frente a val NLL y valor en val por época;
- ``valor_vs_umbral.png``: valor frente a t_L, t_H y umbral único del mejor run, por escenario;
- ``valor_vs_metricas.png``: valor frente a F1 macro y a la tasa ocupado→libre por iteración;
- ``tornado_<escenario>.png`` (+ ``.csv``): sensibilidad ±20 % y ±50 % del mejor run;
- ``valor_vs_costo.png``: valor frente a parámetros, costo de entrenamiento y latencia/tamaño;
- ``matrices_confusion_<split>.png`` y ``matrices_tres_estados_<split>.png``;
- ``metricas_por_camara_<split>.png`` y ``metricas_por_clima_<split>.png``.

Con varias semillas por grupo, las figuras por modelo usan el run de mayor valor comercial
en val del grupo; las de iteraciones usan la media ± desviación de la tabla.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from results_table import (  # noqa: E402
    PRIMARY, SCENARIOS, UNITS, RunRecord, best_record, best_single_threshold, build_table,
    comparable_records, generate, load_records, load_scenarios, operating_rates, param_label,
    record_value, representatives, resolve_dir, source_value,
)

from cupo import paths  # noqa: E402
from cupo.metrics import binary_metrics, breakdown, three_state_rates  # noqa: E402
from cupo.runs import load_npz  # noqa: E402
from cupo.value_model import compute_value, sensitivity, value_curve  # noqa: E402

DPI = 130
EXTRA_SPLITS: tuple[str, ...] = ("test", "ood")
SPLIT_LABELS = {"val": "val", "test": "test", "ood": "OOD (CNRPark A/B)"}
WEATHER_LABELS = {"SUNNY": "Soleado", "OVERCAST": "Nublado", "RAINY": "Lluvia"}
SCENARIO_LABELS = {"comercial": "Comercial", "uvg": "UVG"}
TORNADO_FACTORS: tuple[float, ...] = (0.5, 0.8, 1.2, 1.5)
# Parámetros que entran al modelo solo como producto: sus barras son idénticas.
IDENTICAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("segundos_por_revision", "revisiones_por_hora_amber"),
    ("operadores_incluidos", "fraccion_tiempo_operador"),
)
TORNADO_MAX_ROWS = 18
ZERO_EFFECT = 1e-6  # |Δ| máximo (Q) por debajo del cual un parámetro no tiene efecto
HISTORY_REQUIRED = ("epoch", "val_loss", "val_value")
SUBSET_WARNING = " [solo subconjuntos de datos: no concluyente]"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def setup_style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 100})


def save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")
    return path


def unit(scenario: str) -> str:
    return UNITS.get(scenario, "Q")


def q_formatter(ax: plt.Axes, axis: str = "y") -> None:
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def subset_note(recs: Sequence[RunRecord]) -> str:
    """Aviso para los títulos cuando las figuras se hacen solo con subconjuntos."""
    return SUBSET_WARNING if recs and not any(r.is_full for r in recs) else ""


def labels_for(recs: Sequence[RunRecord]) -> dict[str, str]:
    """Etiqueta corta por grupo: la iteración si es única; si no, iteración + grupo."""
    counts = pd.Series([r.iteration for r in recs]).value_counts()
    return {r.group: r.iteration if counts[r.iteration] == 1 else f"{r.iteration}: {r.group}"
            for r in recs}


def group_colors(groups: Sequence[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab10" if len(groups) <= 10 else "tab20")
    return {g: cmap(i % cmap.N) for i, g in enumerate(groups)}


def split_predictions(rec: RunRecord, split: str) -> pd.DataFrame | None:
    """y, p, cámara y clima de un split; en val se aplica la temperatura del run.

    ``test_preds.npz`` y ``ood_preds.npz`` ya guardan ``p`` con la temperatura del run.
    """
    if split == "val":
        if rec.val is None:
            return None
        data, p = rec.val, rec.val_p()
    else:
        path = rec.run_dir / f"{split}_preds.npz"
        if not path.exists():
            return None
        data = load_npz(path)
        p = np.asarray(data["p"], dtype=float)
    if not {"y", "camera", "weather"} <= set(data):
        print(f"Aviso: {rec.run_id}: {split}_preds sin y/cámara/clima; se omite")
        return None
    return pd.DataFrame({"y": np.asarray(data["y"]).astype(int), "p": p,
                         "camera": np.asarray(data["camera"]).astype(str),
                         "weather": np.asarray(data["weather"]).astype(str)})


# ---------------------------------------------------------------------------
# 1. Curvas de entrenamiento
# ---------------------------------------------------------------------------

def _history(rec: RunRecord) -> pd.DataFrame | None:
    """``history.csv`` con las columnas necesarias; si no sirve, avisa y devuelve None."""
    path = rec.run_dir / "history.csv"
    if not path.exists():
        return None
    try:
        hist = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        print(f"Aviso: {rec.run_id}/history.csv no se puede leer ({type(exc).__name__}); se omite")
        return None
    missing = [c for c in HISTORY_REQUIRED if c not in hist.columns]
    if missing or not len(hist):
        print(f"Aviso: {rec.run_id}/history.csv vacío o sin columnas {missing}; se omite")
        return None
    return hist


def _train_column(hist: pd.DataFrame) -> str | None:
    """NLL de train comparable con val (``train_nll``) o, si no hay, ``train_loss``."""
    return next((c for c in ("train_nll", "train_loss") if c in hist and hist[c].notna().any()), None)


def fig_training_curves(records: Sequence[RunRecord], out_dir: Path) -> Path | None:
    reps = representatives(records)
    hists = {r.group: _history(r) for r in reps}
    reps = [r for r in reps if hists[r.group] is not None]
    if not reps:
        return None
    colors, labels = group_colors([r.group for r in reps]), labels_for(reps)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for rec in reps:
        hist, color = hists[rec.group], colors[rec.group]
        style = "o" if len(hist) == 1 else ""
        axes[0].plot(hist["epoch"], hist["val_loss"], f"{style}-", color=color, label=labels[rec.group])
        train_col = _train_column(hist)
        if train_col:
            axes[0].plot(hist["epoch"], hist[train_col], f"{style}--", color=color, alpha=0.7)
        axes[1].plot(hist["epoch"], hist["val_value"], f"{style}-", color=color, label=labels[rec.group])
    axes[0].set_title("Pérdida (NLL sin pesos): train (- -) frente a val (—)")
    axes[0].set_ylabel("NLL")
    scen = str(reps[0].summary.get("select_scenario", PRIMARY))
    axes[1].set_title(f"Valor en val por época (escenario {scen}, umbrales óptimos)")
    axes[1].set_ylabel(f"Valor ({unit(scen)})")
    q_formatter(axes[1])
    for ax in axes:
        ax.set_xlabel("época")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Curvas de entrenamiento comparadas (un run por modelo)" + subset_note(reps))
    fig.tight_layout()
    return save(fig, out_dir, "curvas_entrenamiento.png")


# ---------------------------------------------------------------------------
# 2. Valor frente a los umbrales (mejor run)
# ---------------------------------------------------------------------------

def value_vs_t_L(y: np.ndarray, p: np.ndarray, cfg: Mapping, t_H: float, step: float = 0.01) -> pd.DataFrame:
    """Valor frente a t_L ∈ [0, t_H] con t_H fijo."""
    grid = np.round(np.arange(0.0, t_H + step / 2, step), 10)
    vals = [compute_value(three_state_rates(y, p, float(t), t_H), cfg)["objective"] for t in grid]
    return pd.DataFrame({"t_L": grid, "value": vals})


def _robust_ylim(ax: plt.Axes, values: Sequence[float], q: float = 10.0) -> None:
    """Recorta el eje y para que las caídas extremas (t → 0 o t → 1) no oculten el máximo."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    lo, hi = np.percentile(vals, q), vals.max()
    if vals.min() < lo:
        pad = 0.08 * max(hi - lo, 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        ax.text(0.99, 0.02, "eje y recortado", transform=ax.transAxes, ha="right",
                fontsize=7, color="dimgray")


def _mark_point(ax: plt.Axes, x: float, label: str, color: str = "tab:red") -> None:
    ax.axvline(x, color=color, ls=":", lw=1.2, label=label)


def _panel_t_L(ax: plt.Axes, y, p, cfg, name: str, t_L: float, t_H: float) -> None:
    by_l = value_vs_t_L(y, p, cfg, t_H)
    ax.plot(by_l["t_L"], by_l["value"], color="tab:blue")
    _mark_point(ax, t_L, f"t_L elegido = {t_L:.2f}")
    _robust_ylim(ax, by_l["value"])
    ax.set_xlabel(f"t_L (con t_H = {t_H:.2f} fijo)")
    ax.set_title(f"{SCENARIO_LABELS.get(name, name)}: valor frente a t_L")


def _panel_t_H(ax: plt.Axes, by_h: pd.DataFrame, name: str, t_H: float) -> None:
    ax.plot(by_h["t_H"], by_h["value"], color="tab:blue", label="valor (t_L óptimo ≤ t_H)")
    _mark_point(ax, t_H, f"t_H elegido = {t_H:.2f}")
    _robust_ylim(ax, by_h["value"])
    twin = ax.twinx()
    twin.plot(by_h["t_H"], 100 * by_h["verify_rate"], color="tab:gray", ls="--", lw=1)
    twin.set_ylabel("por verificar (%)", color="tab:gray")
    twin.grid(False)
    ax.set_xlabel("t_H")
    ax.set_title(f"{SCENARIO_LABELS.get(name, name)}: valor frente a t_H (- - % por verificar)")


def _panel_single(ax: plt.Axes, single: pd.DataFrame, y, p, cfg, name: str) -> None:
    ax.plot(single["t"], single["value"], color="tab:purple")
    _mark_point(ax, 0.5, "t = 0.5", color="tab:gray")
    _robust_ylim(ax, single["value"])
    t_best, v_best = best_single_threshold(y, p, cfg)
    ax.plot(t_best, v_best, "o", color="tab:red", label=f"mejor umbral único t = {t_best:.2f}")
    ax.set_xlabel("umbral único t (sin franja por verificar)")
    ax.set_title(f"{SCENARIO_LABELS.get(name, name)}: decisión de 2 estados")


def fig_value_vs_threshold(rec: RunRecord, scenarios: Mapping[str, Mapping], out_dir: Path) -> Path | None:
    df = split_predictions(rec, "val")
    if df is None:
        return None
    y, p = df["y"].to_numpy(), df["p"].to_numpy()
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(16, 4.2 * len(scenarios)), squeeze=False)
    for row, (name, cfg) in enumerate(scenarios.items()):
        thr = rec.thresholds.get(name) or {"t_L": 0.5, "t_H": 0.5}
        t_L, t_H = float(thr["t_L"]), float(thr["t_H"])
        curves = value_curve(y, p, cfg)
        _panel_t_L(axes[row, 0], y, p, cfg, name, t_L, t_H)
        _panel_t_H(axes[row, 1], curves["by_t_H"], name, t_H)
        _panel_single(axes[row, 2], curves["single_threshold"], y, p, cfg, name)
        for ax in axes[row]:
            ax.set_ylabel(f"Valor en val ({unit(name)})")
            q_formatter(ax)
            ax.legend(fontsize=8, loc="center left")
    fig.suptitle(f"Curva de valor del mejor run en val: {rec.run_id}"
                 + (f" (calibrado, T = {rec.temperature:.3f})" if rec.calibrated else "")
                 + subset_note([rec]))
    fig.tight_layout()
    return save(fig, out_dir, "valor_vs_umbral.png")


# ---------------------------------------------------------------------------
# 3. Valor frente a métricas técnicas a lo largo de las iteraciones
# ---------------------------------------------------------------------------

def iteration_points(table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Filas comparables con modelo (sin B0), en orden de iteración, y una nota para el título.

    Si hay grupos con el conjunto completo se excluyen los subconjuntos; si no, se usan los
    subconjuntos y la nota lo advierte.
    """
    rows = table[table["iteration"] != "B0"]
    full = rows[rows["subset"].astype(float) >= 1]
    if len(full):
        dropped = len(rows) - len(full)
        note = f" ({dropped} grupo(s) con subconjunto excluidos)" if dropped else ""
        return full.reset_index(drop=True), note
    return rows.reset_index(drop=True), (SUBSET_WARNING if len(rows) else "")


def _point_labels(pts: pd.DataFrame) -> list[str]:
    counts = pts["iteration"].value_counts()
    return [it if counts[it] == 1 else f"{it}: {g}" for it, g in zip(pts["iteration"], pts["group"])]


def _scatter_iterations(ax: plt.Axes, pts: pd.DataFrame, xcol: str, ycol: str,
                        x_scale: float = 1.0) -> None:
    x = pts[xcol].astype(float) * x_scale
    y = pts[ycol].astype(float)
    xerr = pts.get(f"{xcol}_std", pd.Series(np.nan, index=pts.index)).astype(float) * x_scale
    yerr = pts.get(f"{ycol}_std", pd.Series(np.nan, index=pts.index)).astype(float)
    ax.plot(x, y, "-", color="lightgray", zorder=1)
    ax.errorbar(x, y, xerr=xerr.fillna(0), yerr=yerr.fillna(0), fmt="o", color="tab:blue",
                ecolor="tab:gray", capsize=3, zorder=2)
    for xi, yi, lab in zip(x, y, _point_labels(pts)):
        if not (math.isnan(xi) or math.isnan(yi)):
            ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
    # B0 vale 0: se anota en lugar de dibujarlo para no aplastar las diferencias entre modelos.
    ax.text(0.01, 0.01, "B0 (sin sistema) = 0, fuera del eje", transform=ax.transAxes,
            fontsize=7.5, color="dimgray", va="bottom")


def fig_value_vs_metrics(table: pd.DataFrame, out_dir: Path) -> Path | None:
    pts, note = iteration_points(table)
    if pts.empty:
        return None
    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(13, 4.5 * len(SCENARIOS)), squeeze=False)
    for row, scen in enumerate(SCENARIOS):
        ycol = f"val_value_{scen}"
        _scatter_iterations(axes[row, 0], pts, "val_f1_macro", ycol)
        axes[row, 0].set_xlabel("F1 macro en val (umbral 0.5)")
        _scatter_iterations(axes[row, 1], pts, "val_rate_busy_as_free", ycol, x_scale=100)
        axes[row, 1].set_xlabel("tasa ocupado→libre en val (%, umbral 0.5)")
        for ax in axes[row]:
            ax.set_ylabel(f"Valor {SCENARIO_LABELS[scen]} en val ({unit(scen)})")
            q_formatter(ax)
    fig.suptitle("Valor de negocio frente a métricas técnicas a lo largo de las iteraciones "
                 "(línea gris: orden de las iteraciones)" + note)
    fig.tight_layout()
    return save(fig, out_dir, "valor_vs_metricas.png")


# ---------------------------------------------------------------------------
# 4. Tornado de sensibilidad (mejor run)
# ---------------------------------------------------------------------------

def merge_identical_pairs(sens: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Une los pares de ``IDENTICAL_PAIRS`` cuando sus Δ coinciden en todos los factores.

    La fila unida queda con la etiqueta en español ``"A = B"``.
    """
    merged = []
    for a, b in IDENTICAL_PAIRS:
        da = sens.loc[sens["param"] == a].sort_values("factor")
        db = sens.loc[sens["param"] == b].sort_values("factor")
        if da.empty or db.empty or len(da) != len(db):
            continue
        same = np.allclose(da["delta"].to_numpy(), db["delta"].to_numpy(), rtol=1e-9, atol=1e-6)
        if same and da["clipped"].to_numpy().tolist() == db["clipped"].to_numpy().tolist():
            label = f"{param_label(a)} = {param_label(b)}"
            sens = sens.loc[sens["param"] != b].copy()
            sens.loc[sens["param"] == a, "param"] = label
            merged.append(label)
    return sens, merged


def tornado_data(sens: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """(tabla ordenada por efecto, pares unidos, parámetros sin efecto), con nombres en español."""
    sens, merged = merge_identical_pairs(sens.copy())
    singles = ~sens["param"].isin(merged)
    sens.loc[singles, "param"] = sens.loc[singles, "param"].map(param_label)
    effect = sens.groupby("param")["delta"].apply(lambda d: float(np.abs(d).max()))
    zero = sorted(effect[effect <= ZERO_EFFECT].index)
    order = effect[effect > ZERO_EFFECT].sort_values(ascending=False).index.tolist()
    data = sens[sens["param"].isin(order)].copy()
    data["param"] = pd.Categorical(data["param"], categories=order, ordered=True)
    return data.sort_values(["param", "factor"]), merged, zero


def _tornado_bar(ax: plt.Axes, ypos: float, row: pd.Series) -> None:
    factor = float(row["factor"])
    outer = abs(factor - 1) >= 0.45
    color = "tab:blue" if factor < 1 else "tab:orange"
    ax.barh(ypos, row["delta"], height=0.75 if outer else 0.4, color=color,
            alpha=0.35 if outer else 0.9, hatch="//" if row["clipped"] else None,
            edgecolor="black" if row["clipped"] else color, lw=0.6)


TORNADO_HANDLES = (
    ("tab:blue", 0.35, "−50 %"), ("tab:blue", 0.9, "−20 %"),
    ("tab:orange", 0.9, "+20 %"), ("tab:orange", 0.35, "+50 %"),
)


def _tornado_notes(rec: RunRecord, name: str, base: float, merged: list[str], zero: list[str],
                   n_params: int) -> list[str]:
    notes = []
    stored = source_value(rec, name)[0]
    if not math.isnan(stored) and not math.isclose(stored, base, rel_tol=1e-6, abs_tol=1e-6):
        notes.append(f"El valor base se recalculó con el modelo de valor actual; el run guardó "
                     f"Q {stored:,.0f} (versión anterior de configs/value o value_model.py)")
    if merged:
        notes.append("Barras idénticas (entran al modelo como producto): " + "; ".join(merged))
    if zero:
        notes.append("Sin efecto en el objetivo: " + ", ".join(zero))
    if n_params > TORNADO_MAX_ROWS:
        notes.append(f"Se muestran los {TORNADO_MAX_ROWS} supuestos de mayor efecto; el resto en el CSV")
    return notes


def fig_tornado(rec: RunRecord, name: str, cfg: Mapping, out_dir: Path) -> Path | None:
    rates = operating_rates(rec, name)
    if rates is None:
        return None
    sens = sensitivity(rates, cfg, factors=TORNADO_FACTORS)
    sens.assign(param_label=sens["param"].map(param_label)).to_csv(
        out_dir / f"tornado_{name}.csv", index=False, encoding="utf-8-sig")
    data, merged, zero = tornado_data(sens)
    params = list(data["param"].cat.categories)[:TORNADO_MAX_ROWS]
    if not params:
        return None
    base = float(sens["base_objective"].iloc[0])
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(params) + 2.6))
    ylabels = []
    for i, param in enumerate(params):
        rows = data[data["param"] == param]
        # Primero las barras de ±50 % y encima las de ±20 %.
        for _, r in rows.iloc[np.argsort(-np.abs(rows["factor"].to_numpy() - 1), kind="stable")].iterrows():
            _tornado_bar(ax, -i, r)
        star = " *" if rows["clipped"].any() else ""
        ylabels.append(f"{param}: base {rows['base_param_value'].iloc[0]:,.4g}{star}")
    ax.set_yticks([-i for i in range(len(params))], ylabels, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Δ valor frente al caso base ({unit(name)})")
    q_formatter(ax, "x")
    handles = [Patch(color=c, alpha=a, label=lab) for c, a, lab in TORNADO_HANDLES]
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="//",
                         label="* recortado a su rango válido"))
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_title(f"Sensibilidad del valor {SCENARIO_LABELS.get(name, name)} ({rec.run_id}; "
                 f"t_L = {rates['t_L']:.2f}, t_H = {rates['t_H']:.2f} fijos)\n"
                 f"valor base = Q {base:,.0f} ({unit(name)}); cada supuesto ×0.5, ×0.8, ×1.2, ×1.5"
                 + subset_note([rec]))
    notes = _tornado_notes(rec, name, base, merged, zero, len(data["param"].cat.categories))
    if notes:
        fig.text(0.01, -0.01, "\n".join(notes), fontsize=7.5, va="top", wrap=True)
    fig.tight_layout()
    return save(fig, out_dir, f"tornado_{name}.png")


# ---------------------------------------------------------------------------
# 5. Valor frente a costo y tamaño (rendimientos decrecientes)
# ---------------------------------------------------------------------------

COST_AXES: tuple[tuple[str, str], ...] = (
    ("params_total", "parámetros (escala log)"),
    ("train_cost_q", "costo de entrenamiento por run (Q, escala log)"),
    ("latency_ms", "latencia CPU p50 (ms/espacio, escala log)"),
    ("size_mb", "tamaño del modelo (MB, escala log)"),
)


def fig_value_vs_cost(table: pd.DataFrame, out_dir: Path) -> Path | None:
    pts, note = iteration_points(table)
    if pts.empty:
        return None
    axes_cfg = [(c, lab) for c, lab in COST_AXES
                if c in pts and (pts[c].astype(float) > 0).any()][:3]
    if not axes_cfg:
        return None
    fig, axes = plt.subplots(1, len(axes_cfg), figsize=(5.5 * len(axes_cfg), 4.5), squeeze=False)
    ycol = f"val_value_{PRIMARY}"
    for ax, (col, label) in zip(axes[0], axes_cfg):
        sub = pts[pts[col].astype(float) > 0].sort_values(col)
        _scatter_iterations(ax, sub, col, ycol)
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel(f"Valor comercial en val ({unit(PRIMARY)})")
        q_formatter(ax)
    fig.suptitle("Rendimientos decrecientes: valor frente a tamaño y costo "
                 "(línea gris: orden por el eje x)" + note)
    fig.tight_layout()
    return save(fig, out_dir, "valor_vs_costo.png")


# ---------------------------------------------------------------------------
# 6. Matrices de confusión
# ---------------------------------------------------------------------------

def _grid(n: int, ncols: int = 4) -> tuple[int, int]:
    ncols = max(1, min(ncols, n))
    return math.ceil(n / ncols), ncols


def _draw_matrix(ax: plt.Axes, counts: np.ndarray, xlabels: Sequence[str], title: str) -> None:
    totals = counts.sum(axis=1, keepdims=True)
    frac = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for (i, j), c in np.ndenumerate(counts):
        ax.text(j, i, f"{100 * frac[i, j]:.1f} %\n({int(c):,})", ha="center", va="center",
                fontsize=8, color="white" if frac[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=8)
    ax.set_yticks([0, 1], ["real libre", "real ocupado"], fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(False)


def three_state_counts(df: pd.DataFrame, t_L: float, t_H: float) -> np.ndarray:
    """Conteos [real libre, real ocupado] × [mostrado libre, por verificar, ocupado]."""
    p, y = df["p"].to_numpy(), df["y"].to_numpy()
    state = np.where(p < t_L, 0, np.where(p >= t_H, 2, 1))
    return np.array([[np.sum((y == k) & (state == s)) for s in range(3)] for k in (0, 1)])


def _confusion_figure(items: Sequence[tuple[RunRecord, pd.DataFrame]], kind: str, split: str,
                      out_dir: Path) -> Path:
    labels = labels_for([r for r, _ in items])
    nrows, ncols = _grid(len(items))
    width = 3.6 * ncols + (1.2 if kind != "binaria" else 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 3.3 * nrows), squeeze=False)
    for ax in axes.flat[len(items):]:
        ax.axis("off")
    for ax, (rec, df) in zip(axes.flat, items):
        if kind == "binaria":
            counts = np.asarray(binary_metrics(df["y"], df["p"], 0.5)["confusion"])
            _draw_matrix(ax, counts, ["pred. libre", "pred. ocupado"], labels[rec.group])
        else:
            thr = rec.thresholds.get(PRIMARY) or {"t_L": 0.5, "t_H": 0.5}
            counts = three_state_counts(df, float(thr["t_L"]), float(thr["t_H"]))
            _draw_matrix(ax, counts, ["libre", "verificar", "ocupado"],
                         f"{labels[rec.group]} (t_L={thr['t_L']:.2f}, t_H={thr['t_H']:.2f})")
    title = ("Matrices de confusión con umbral 0.5" if kind == "binaria" else
             "Decisión de tres estados en el punto de operación comercial")
    fig.suptitle(f"{title} — {SPLIT_LABELS[split]} (% por fila real)"
                 + subset_note([r for r, _ in items]))
    fig.tight_layout()
    name = "matrices_confusion" if kind == "binaria" else "matrices_tres_estados"
    return save(fig, out_dir, f"{name}_{split}.png")


def fig_confusions(reps: Sequence[RunRecord], split: str, out_dir: Path) -> list[Path]:
    items = [(r, split_predictions(r, split)) for r in reps]
    items = [(r, df) for r, df in items if df is not None]
    if not items:
        return []
    return [_confusion_figure(items, kind, split, out_dir) for kind in ("binaria", "tres_estados")]


# ---------------------------------------------------------------------------
# 7. Métricas por cámara y por clima
# ---------------------------------------------------------------------------

def breakdown_frame(reps: Sequence[RunRecord], split: str, by: str) -> pd.DataFrame:
    """Filas (grupo, categoría, n, f1_macro, rate_busy_as_free) por run representativo."""
    rows = []
    for rec in reps:
        df = split_predictions(rec, split)
        if df is None:
            continue
        for cat, m in breakdown(df, by=by).items():
            rows.append({"group": rec.group, "category": cat, "n": m["n"],
                         "f1_macro": m["f1_macro"], "rate_busy_as_free": m["rate_busy_as_free"]})
    return pd.DataFrame(rows)


def _category_order(cats: Sequence[str], by: str) -> list[str]:
    if by == "weather":
        known = [c for c in WEATHER_LABELS if c in cats]
        return known + sorted(c for c in cats if c not in known)
    return sorted(cats, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c))


def _n_text(data: pd.DataFrame, category: str) -> str:
    """'n=…' de una categoría; si los modelos se evaluaron con conjuntos distintos, el rango."""
    ns = data.loc[data["category"] == category, "n"].astype(int)
    lo, hi = int(ns.min()), int(ns.max())
    return f"n={lo:,}" if lo == hi else f"n={lo:,}–{hi:,}"


def _category_label(cat: str, by: str, data: pd.DataFrame) -> str:
    name = WEATHER_LABELS.get(cat, cat) if by == "weather" else f"cám. {cat}"
    return f"{name}\n({_n_text(data, cat)})"


def fig_breakdown(reps: Sequence[RunRecord], split: str, by: str, out_dir: Path) -> Path | None:
    data = breakdown_frame(reps, split, by)
    if data.empty:
        return None
    groups = list(dict.fromkeys(data["group"]))
    present = [r for r in reps if r.group in groups]
    labels, colors = labels_for(present), group_colors(groups)
    cats = _category_order(list(data["category"].unique()), by)
    width = 0.8 / len(groups)
    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 1.3 * len(cats) + 3), 8.5))
    for k, g in enumerate(groups):
        sub = data[data["group"] == g].set_index("category").reindex(cats)
        xpos = np.arange(len(cats)) + (k - (len(groups) - 1) / 2) * width
        axes[0].bar(xpos, sub["f1_macro"], width, color=colors[g], label=labels[g])
        axes[1].bar(xpos, 100 * sub["rate_busy_as_free"], width, color=colors[g], label=labels[g])
    axes[0].set_ylabel("F1 macro (umbral 0.5)")
    lo = np.nanmin(data["f1_macro"].to_numpy())
    axes[0].set_ylim(max(0.0, lo - 0.05), 1.0)
    axes[1].set_ylabel("tasa ocupado→libre (%, umbral 0.5)")
    for ax in axes:
        ax.set_xticks(np.arange(len(cats)), [_category_label(c, by, data) for c in cats], fontsize=8)
    axes[0].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    what = "clima" if by == "weather" else "cámara"
    rain = " — val no tiene días de lluvia" if split == "val" and by == "weather" and "RAINY" not in cats else ""
    fig.suptitle(f"Métricas por {what} en {SPLIT_LABELS[split]}{rain}{subset_note(present)}")
    fig.tight_layout()
    name = "metricas_por_clima" if by == "weather" else "metricas_por_camara"
    return save(fig, out_dir, f"{name}_{split}.png")


# ---------------------------------------------------------------------------
# Orquestación y CLI
# ---------------------------------------------------------------------------

def available_splits(reps: Sequence[RunRecord]) -> list[str]:
    """val y, solo si ya existen sus predicciones, test y OOD."""
    extra = [sp for sp in EXTRA_SPLITS if any((r.run_dir / f"{sp}_preds.npz").exists() for r in reps)]
    return ["val", *extra]


def _safe(label: str, fn: Callable[..., Any], *args: Any) -> list[Path]:
    """Ejecuta una figura; si falla, avisa y sigue (una figura rota no frena las demás)."""
    try:
        out = fn(*args)
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        print(f"Aviso: no se pudo generar {label}: {type(exc).__name__}: {exc}")
        return []
    if out is None:
        return []
    return list(out) if isinstance(out, list) else [out]


def make_all(records: Sequence[RunRecord], out_dir: Path) -> list[Path]:
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios()
    comparable_records(records, warn=True)
    table = build_table(records, scenarios)
    reps = representatives(records)
    best = best_record(records)
    made = _safe("curvas de entrenamiento", fig_training_curves, records, out_dir)
    made += _safe("valor frente a métricas", fig_value_vs_metrics, table, out_dir)
    made += _safe("valor frente a costo", fig_value_vs_cost, table, out_dir)
    if best is not None:
        print(f"Mejor run (valor comercial en val): {best.run_id} = Q {record_value(best):,.0f}/mes")
        made += _safe("valor frente al umbral", fig_value_vs_threshold, best, scenarios, out_dir)
        for name, cfg in scenarios.items():
            made += _safe(f"tornado {name}", fig_tornado, best, name, cfg, out_dir)
    for split in available_splits(reps):
        made += _safe(f"matrices de confusión ({split})", fig_confusions, reps, split, out_dir)
        for by in ("camera", "weather"):
            made += _safe(f"métricas por {by} ({split})", fig_breakdown, reps, split, by, out_dir)
    return made


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera las figuras de resultados de los runs.")
    parser.add_argument("--runs-dir", default=None, help="carpeta de runs (por defecto runs/)")
    parser.add_argument("--include-smoke", action="store_true",
                        help="incluye también runs/_smoke/ (pruebas de humo)")
    parser.add_argument("--out", default=None, help="carpeta de salida (por defecto results/figures/)")
    parser.add_argument("--with-table", action="store_true",
                        help="genera también la tabla de iteraciones (results_table.py)")
    parser.add_argument("--table-out", default=None,
                        help="carpeta de la tabla con --with-table (por defecto results/)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs_dir = resolve_dir(args.runs_dir, paths.runs_dir())
    out_dir = resolve_dir(args.out, paths.results_dir() / "figures")
    records = load_records(runs_dir, args.include_smoke)
    if args.with_table:
        generate(runs_dir, resolve_dir(args.table_out, paths.results_dir()), args.include_smoke,
                 records=records)
    if not records:
        print(f"No hay runs completos en {runs_dir}; no se generan figuras.")
        return 1
    made = make_all(records, out_dir)
    print(f"{len(made)} figura(s) en {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
