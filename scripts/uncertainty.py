"""Incertidumbre (IC95 por bootstrap) de las métricas y del valor de negocio en test y OOD.

Uso:
    python scripts/uncertainty.py --allow-test [--runs-dir runs] [--out results]
                                  [--n-boot N] [--seed 0] [--final mnv3_ft_cost]
                                  [--refs b1_hog_lr b2_malexnet] [--iterations results/iteraciones.csv]

Solo reporta: lee las predicciones que el congelamiento dejó en cada run
(``test_preds.npz`` y ``ood_preds.npz``) junto con la temperatura y los umbrales congelados
del run (``runs.operating_point``: ``calibration.json`` si existe, si no ``thresholds.json``).
No re-optimiza umbrales ni temperatura y no necesita el checkpoint del modelo.

Para cada grupo de runs (``name`` de la config) y cada partición (test, OOD) calcula, con las
mismas funciones y la misma semilla que ``cupo.evaluate``:
- F1 macro y tasa ocupado→libre con umbral 0.5 (``CI_METRICS`` + ``metrics.bootstrap_ci``);
- valor comercial (Q/mes) y UVG (Q/semestre) en el punto de operación (``value_report``).
Así la estimación puntual coincide con la de ``results/iteraciones.csv``, que se lee solo para
verificarlo. Con varias semillas se reporta media ± desviación y el IC95 es el promedio de los
IC95 de cada semilla (igual que la tabla de iteraciones).

Además, la diferencia de valor del modelo final frente a cada referencia (B1, B2) con un
bootstrap pareado: las mismas imágenes remuestreadas para los dos modelos (emparejadas por
``path``), con su IC95 y la proporción de remuestras en que el final supera a la referencia.

Escribe ``incertidumbre.csv`` (formato largo) e ``incertidumbre.md`` (tablas para el informe).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from results_table import (  # noqa: E402
    UNITS,
    _md_table,
    fmt_num,
    iteration_id,
    iteration_rank,
    resolve_dir,
    subset_of,
)

from cupo import metrics as M  # noqa: E402
from cupo import paths  # noqa: E402
from cupo.evaluate import (  # noqa: E402
    CI_METRICS,
    DEFAULT_N_BOOT,
    load_scenarios,
    probs_from_logits,
    value_report,
)
from cupo.runs import load_json, load_npz, load_yaml, operating_point  # noqa: E402
from cupo.value_model import compute_value  # noqa: E402

SPLITS: tuple[str, ...] = ("test", "ood")
SPLIT_NAMES = {"test": "Test: CNR-EXT, fechas de test", "ood": "OOD: CNRPark A/B"}
SCENARIOS: tuple[str, ...] = ("comercial", "uvg")
BINARY_METRICS: tuple[str, ...] = ("f1_macro", "rate_busy_as_free")
METRICS: tuple[str, ...] = (*BINARY_METRICS, *(f"value_{s}" for s in SCENARIOS))
DEFAULT_FINAL = "mnv3_ft_cost"
DEFAULT_REFS: tuple[str, ...] = ("b1_hog_lr", "b2_malexnet")
DEFAULT_SEED = 0  # la misma semilla que usa cupo.evaluate en el congelamiento
OUT_NAME = "incertidumbre"
REL_TOL = 1e-6  # tolerancia para comparar con results/iteraciones.csv
ABS_TOL = 1e-9

CSV_COLUMNS: tuple[str, ...] = (
    "kind", "iteration", "group", "reference", "split", "metric", "estimate", "estimate_std",
    "ci_lo", "ci_hi", "p_better", "n", "n_runs", "n_boot", "seed", "iteraciones_value",
    "iteraciones_std", "matches_iteraciones", "run_ids",
)


# ---------------------------------------------------------------------------
# Lectura de runs (sin modelo ni checkpoint)
# ---------------------------------------------------------------------------

@dataclass
class SplitPreds:
    """Etiquetas, P(ocupado) con la temperatura del run y ruta de cada imagen."""

    y: np.ndarray
    p: np.ndarray
    path: np.ndarray


@dataclass
class RunPreds:
    """Un run con su punto de operación congelado y sus predicciones en test y OOD."""

    run_dir: Path
    name: str
    temperature: float
    thresholds: dict[str, dict[str, float]]
    n_boot: int
    splits: dict[str, SplitPreds]

    @property
    def run_id(self) -> str:
        return self.run_dir.name


def load_split_preds(run_dir: Path, split: str, temperature: float) -> SplitPreds:
    """``<split>_preds.npz`` con p = softmax(logits / T)[:, 1], como en ``cupo.evaluate``."""
    data = load_npz(run_dir / f"{split}_preds.npz")
    p = probs_from_logits(data["logits"], temperature)
    return SplitPreds(y=data["y"].astype(int), p=p, path=data["path"].astype(str))


def is_candidate(run_dir: Path) -> bool:
    """Carpeta de un run completo con predicciones de test y OOD (sin ``_smoke`` ni ``_obsoleto``)."""
    if run_dir.name.startswith("_") or not (run_dir / "config.yaml").is_file():
        return False
    return all((run_dir / f"{sp}_preds.npz").is_file() for sp in SPLITS)


def load_run_preds(run_dir: Path, n_boot: int | None) -> RunPreds | None:
    """Run listo para el bootstrap; ``None`` si usó un subconjunto o no tiene umbrales."""
    config = load_yaml(run_dir / "config.yaml")
    if subset_of(config.get("data", {}).get("subset")) < 1:
        print(f"Se omite {run_dir.name}: entrenado con un subconjunto de datos")
        return None
    temperature, thresholds, _ = operating_point(run_dir)
    if not thresholds:
        print(f"Se omite {run_dir.name}: sin umbrales congelados")
        return None
    boots = n_boot or int(config.get("eval", {}).get("n_boot", DEFAULT_N_BOOT))
    splits = {sp: load_split_preds(run_dir, sp, temperature) for sp in SPLITS}
    return RunPreds(run_dir=run_dir, name=str(config.get("name") or run_dir.name),
                    temperature=temperature, thresholds=thresholds, n_boot=boots, splits=splits)


def load_groups(runs_dir: Path, n_boot: int | None) -> dict[str, list[RunPreds]]:
    """Runs agrupados por ``name``, en el orden de iteraciones de la tabla."""
    groups: dict[str, list[RunPreds]] = {}
    for run_dir in sorted(d for d in runs_dir.iterdir() if d.is_dir() and is_candidate(d)):
        run = load_run_preds(run_dir, n_boot)
        if run is not None:
            groups.setdefault(run.name, []).append(run)
    order = sorted(groups, key=lambda g: (iteration_rank(iteration_id(g)), iteration_id(g), g))
    return {g: groups[g] for g in order}


# ---------------------------------------------------------------------------
# Emparejamiento por imagen
# ---------------------------------------------------------------------------

def alignment(ref_paths: np.ndarray, paths_: np.ndarray, where: str = "") -> np.ndarray:
    """Índices que reordenan ``paths_`` al orden de ``ref_paths``; exige el mismo conjunto."""
    tag = f" ({where})" if where else ""
    if len(set(ref_paths)) != len(ref_paths) or len(set(paths_)) != len(paths_):
        raise ValueError(f"Hay rutas repetidas en las predicciones{tag}: no se puede emparejar por path")
    if set(ref_paths) != set(paths_):
        raise ValueError(f"Los runs no tienen el mismo conjunto de imágenes{tag}")
    pos = {p: i for i, p in enumerate(paths_)}
    return np.array([pos[p] for p in ref_paths], dtype=int)


def check_same_images(groups: Mapping[str, Sequence[RunPreds]]) -> dict[str, int]:
    """Verifica que todos los runs tengan las mismas imágenes y etiquetas en cada partición."""
    runs = [r for rs in groups.values() for r in rs]
    out: dict[str, int] = {}
    for sp in SPLITS:
        ref = runs[0].splits[sp]
        for run in runs[1:]:
            cur = run.splits[sp]
            idx = alignment(ref.path, cur.path, f"{run.run_id} [{sp}] frente a {runs[0].run_id}")
            if not np.array_equal(ref.y, cur.y[idx]):
                raise ValueError(f"{run.run_id} [{sp}]: etiquetas distintas para las mismas imágenes")
        out[sp] = int(ref.y.size)
    return out


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def objective_fn(cfg: Mapping[str, Any], t_L: float, t_H: float) -> Callable[[np.ndarray, np.ndarray], float]:
    """Valor del escenario con umbrales fijos, como el objetivo de ``value_report``."""
    def objective(y: np.ndarray, p: np.ndarray) -> float:
        return float(compute_value(M.three_state_rates(y, p, t_L, t_H), cfg)["objective"])
    return objective


def run_estimates(run: RunPreds, split: str, scenarios: Mapping[str, Mapping],
                  seed: int) -> dict[str, tuple[float, float, float]]:
    """``{métrica: (estimación, IC95 inferior, IC95 superior)}`` de un run en una partición."""
    sp = run.splits[split]
    out: dict[str, tuple[float, float, float]] = {}
    for name in BINARY_METRICS:
        fn = CI_METRICS[name]
        lo, hi = M.bootstrap_ci(fn, sp.y, sp.p, n_boot=run.n_boot, seed=seed)
        out[name] = (fn(sp.y, sp.p), lo, hi)
    values = value_report(sp.y, sp.p, run.thresholds, scenarios, run.n_boot, seed)
    for name, v in values.items():
        lo, hi = v["ci95"]["objective"]
        out[f"value_{name}"] = (float(v["objective_value"]), float(lo), float(hi))
    return out


def paired_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                 fn_a: Callable[[np.ndarray, np.ndarray], float],
                 fn_b: Callable[[np.ndarray, np.ndarray], float],
                 n_boot: int, seed: int, alpha: float = 0.05) -> dict[str, float]:
    """Δ = fn_a − fn_b con bootstrap pareado (mismos índices para los dos modelos).

    Los índices se generan igual que en ``metrics.bootstrap_ci`` (misma semilla), así que cada
    remuestra coincide con la usada en el IC95 marginal de cada modelo.
    """
    rng = np.random.default_rng(seed)
    n = y.size
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = fn_a(y[idx], p_a[idx]) - fn_b(y[idx], p_b[idx])
    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": fn_a(y, p_a) - fn_b(y, p_b), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_better": float(np.mean(deltas > 0))}


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------

def _std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def group_rows(group: str, runs: Sequence[RunPreds], scenarios: Mapping[str, Mapping],
               seed: int) -> list[dict[str, Any]]:
    """Filas por partición y métrica: media ± desviación entre semillas y promedio de los IC95."""
    rows = []
    for split in SPLITS:
        per_run = [run_estimates(r, split, scenarios, seed) for r in runs]
        for metric in METRICS:
            if any(metric not in est for est in per_run):
                continue
            est, lo, hi = (np.array([e[metric][k] for e in per_run]) for k in range(3))
            rows.append({
                "kind": "metric", "iteration": iteration_id(group), "group": group,
                "reference": "", "split": split, "metric": metric,
                "estimate": float(est.mean()), "estimate_std": _std(est.tolist()),
                "ci_lo": float(lo.mean()), "ci_hi": float(hi.mean()), "p_better": float("nan"),
                "n": int(runs[0].splits[split].y.size), "n_runs": len(runs),
                "n_boot": runs[0].n_boot, "seed": seed,
                "run_ids": "; ".join(r.run_id for r in runs),
            })
    return rows


def delta_rows(final: RunPreds, ref: RunPreds, scenarios: Mapping[str, Mapping],
               seed: int) -> list[dict[str, Any]]:
    """Δ valor (final − referencia) por partición y escenario con bootstrap pareado."""
    rows = []
    for split in SPLITS:
        a, b = final.splits[split], ref.splits[split]
        idx = alignment(a.path, b.path)
        for scen in SCENARIOS:
            if scen not in final.thresholds or scen not in ref.thresholds:
                continue
            cfg = scenarios[scen]
            fn_a = objective_fn(cfg, final.thresholds[scen]["t_L"], final.thresholds[scen]["t_H"])
            fn_b = objective_fn(cfg, ref.thresholds[scen]["t_L"], ref.thresholds[scen]["t_H"])
            res = paired_delta(a.y, a.p, b.p[idx], fn_a, fn_b, final.n_boot, seed)
            rows.append({
                "kind": "delta", "iteration": iteration_id(final.name), "group": final.name,
                "reference": ref.name, "split": split, "metric": f"value_{scen}",
                "estimate_std": float("nan"), **res, "n": int(a.y.size), "n_runs": 1,
                "n_boot": final.n_boot, "seed": seed,
                "run_ids": f"{final.run_id}; {ref.run_id}",
            })
    return rows


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _close(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def check_iterations(table: pd.DataFrame, iterations_csv: Path) -> pd.DataFrame:
    """Compara cada estimación puntual con la columna ``<split>_<metric>`` de iteraciones.csv."""
    table = table.copy()
    for col in ("iteraciones_value", "iteraciones_std"):
        table[col] = float("nan")
    table["matches_iteraciones"] = pd.NA
    if not iterations_csv.is_file():
        print(f"Aviso: no existe {iterations_csv}; se omite la verificación")
        return table
    it = pd.read_csv(iterations_csv).set_index("group")
    for i, row in table[table["kind"] == "metric"].iterrows():
        col = f"{row['split']}_{row['metric']}"
        if col not in it.columns or row["group"] not in it.index:
            continue
        std_col = f"{col}_std"
        ref = _num(it.at[row["group"], col])
        ref_std = _num(it.at[row["group"], std_col]) if std_col in it.columns else float("nan")
        if math.isnan(ref):
            continue
        ok = _close(row["estimate"], ref) and _close(_num(row["estimate_std"]), ref_std)
        table.loc[i, ["iteraciones_value", "iteraciones_std"]] = [ref, ref_std]
        table.loc[i, "matches_iteraciones"] = bool(ok)
        if not ok:
            print(f"Aviso: {row['group']} {col}: {row['estimate']:.6f} ± {row['estimate_std']} "
                  f"vs iteraciones.csv {ref:.6f} ± {ref_std}")
    return table


def build_table(groups: Mapping[str, Sequence[RunPreds]], final: str, refs: Sequence[str],
                scenarios: Mapping[str, Mapping], seed: int, iterations_csv: Path) -> pd.DataFrame:
    rows = [r for g, runs in groups.items() for r in group_rows(g, runs, scenarios, seed)]
    for ref in refs:
        rows += delta_rows(single_run(groups, final), single_run(groups, ref), scenarios, seed)
    table = check_iterations(pd.DataFrame(rows), iterations_csv)
    return table[list(CSV_COLUMNS)]


def single_run(groups: Mapping[str, Sequence[RunPreds]], name: str) -> RunPreds:
    """El único run de un grupo (el Δ pareado se hace entre runs, no entre promedios)."""
    if name not in groups:
        raise KeyError(f"No hay runs de {name!r} con predicciones de test y OOD")
    if len(groups[name]) != 1:
        raise ValueError(f"{name!r} tiene {len(groups[name])} runs; el Δ pareado necesita uno")
    return groups[name][0]


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

METRIC_FORMAT: dict[str, tuple[int, bool]] = {
    "f1_macro": (4, False), "rate_busy_as_free": (2, True),
    "value_comercial": (0, False), "value_uvg": (0, False),
}


def fmt_estimate(row: Mapping[str, Any]) -> str:
    """'estimación [lo; hi]' o 'media ± desviación [lo; hi]' con varias semillas."""
    dec, pct = METRIC_FORMAT[row["metric"]]
    text = fmt_num(row["estimate"], dec, pct)
    if not math.isnan(_num(row["estimate_std"])):
        text += f" ± {fmt_num(row['estimate_std'], dec, pct)}"
    return f"{text} [{fmt_num(row['ci_lo'], dec, pct)}; {fmt_num(row['ci_hi'], dec, pct)}]"


def fmt_delta(row: Mapping[str, Any]) -> str:
    return (f"{row['estimate']:+,.0f} [{row['ci_lo']:+,.0f}; {row['ci_hi']:+,.0f}]")


def group_label(group: str, final: str) -> str:
    it = iteration_id(group)
    label = f"**{it}**" if it == group else f"**{it}** ({group})"
    return label + (" — modelo final" if group == final else "")


def split_table(table: pd.DataFrame, split: str, final: str) -> list[str]:
    headers = ["ID", "F1 macro [IC95]", "Ocupado→libre [IC95]",
               f"Valor comercial ({UNITS['comercial']}) [IC95]", f"Valor UVG ({UNITS['uvg']}) [IC95]"]
    sub = table[(table["kind"] == "metric") & (table["split"] == split)]
    body = []
    for group, g in sub.groupby("group", sort=False):
        cells = {r["metric"]: fmt_estimate(r) for _, r in g.iterrows()}
        body.append([group_label(group, final), *(cells.get(m, "–") for m in METRICS)])
    return _md_table(headers, body)


def delta_table(table: pd.DataFrame) -> list[str]:
    headers = ["Comparación", "Partición", f"Δ valor comercial ({UNITS['comercial']}) [IC95]",
               "Final > ref. (comercial)", f"Δ valor UVG ({UNITS['uvg']}) [IC95]", "Final > ref. (UVG)"]
    sub = table[table["kind"] == "delta"]
    body = []
    for (ref, split), g in sub.groupby(["reference", "split"], sort=False):
        cells = []
        for scen in SCENARIOS:
            r = g[g["metric"] == f"value_{scen}"]
            cells += (["–", "–"] if r.empty else
                      [fmt_delta(r.iloc[0]), fmt_num(r.iloc[0]["p_better"], 1, pct=True)])
        label = f"{iteration_id(g.iloc[0]['group'])} − {iteration_id(ref)}"
        body.append([label, "Test" if split == "test" else "OOD", *cells])
    return _md_table(headers, body)


def verification_lines(table: pd.DataFrame, iterations_csv: Path) -> list[str]:
    checked = table["matches_iteraciones"].dropna()
    if checked.empty:
        return [f"- No se pudo verificar: no hay columnas de test/OOD en `{iterations_csv.name}`."]
    ok = int(checked.astype(bool).sum())
    lines = [f"- {ok} de {len(checked)} estimaciones puntuales (F1 macro, ocupado→libre y valor "
             f"comercial en test y OOD; media y desviación en grupos con varias semillas) coinciden "
             f"con `{iterations_csv.name}` (tolerancia relativa {REL_TOL:g})."]
    bad = table[table["matches_iteraciones"].eq(False)]
    lines += [f"- No coincide: {r['group']} {r['split']}_{r['metric']} = {r['estimate']:.6f} "
              f"(tabla: {r['iteraciones_value']:.6f})" for _, r in bad.iterrows()]
    return lines


def _row(table: pd.DataFrame, kind: str, group: str, split: str, metric: str,
         reference: str = "") -> pd.Series | None:
    sel = table[(table["kind"] == kind) & (table["group"] == group) & (table["split"] == split)
                & (table["metric"] == metric) & (table["reference"].fillna("") == reference)]
    return None if sel.empty else sel.iloc[0]


def _q(x: float) -> str:
    return f"Q{x:,.0f}" if x >= 0 else f"−Q{-x:,.0f}"


def _overlap_sentence(table: pd.DataFrame, final: str, ref: str, split: str) -> str | None:
    a = _row(table, "metric", final, split, "value_comercial")
    b = _row(table, "metric", ref, split, "value_comercial")
    d = _row(table, "delta", final, split, "value_comercial", ref)
    if a is None or b is None or d is None:
        return None
    overlap = a["ci_lo"] <= b["ci_hi"] and b["ci_lo"] <= a["ci_hi"]
    excludes_zero = d["ci_lo"] > 0 or d["ci_hi"] < 0
    where = "En OOD (CNRPark A/B)" if split == "ood" else "En test"
    fin, rid = iteration_id(final), iteration_id(ref)
    return (f"- {where}, los IC95 del valor comercial del modelo final ({fin}) "
            f"[{_q(a['ci_lo'])}; {_q(a['ci_hi'])}] y de {rid} [{_q(b['ci_lo'])}; {_q(b['ci_hi'])}] "
            f"{'se solapan' if overlap else 'no se solapan'}. La diferencia pareada es "
            f"Δ = {_q(d['estimate'])} al mes [IC95 {_q(d['ci_lo'])}; {_q(d['ci_hi'])}], "
            f"{'que excluye' if excludes_zero else 'que incluye'} el 0; {fin} supera a {rid} en "
            f"{100 * d['p_better']:.1f} % de las remuestras.")


def _seeds_sentence(table: pd.DataFrame) -> str | None:
    multi = table[(table["kind"] == "metric") & (table["n_runs"] > 1)
                  & (table["metric"] == "value_comercial")]
    if multi.empty:
        return None
    lines = []
    for group, rows in multi.groupby("group", sort=False):
        parts = []
        for _, r in rows.iterrows():
            half = (r["ci_hi"] - r["ci_lo"]) / 2
            where = "OOD" if r["split"] == "ood" else "test"
            parts.append(f"en {where} la desviación entre semillas ({_q(r['estimate_std'])}) es "
                         f"{r['estimate_std'] / half:.1f} veces la mitad del IC95 bootstrap ({_q(half)})")
        lines.append(f"- {group}: {'; '.join(parts)}.")
    return ("\n".join(lines) + " Cuando ese cociente se acerca o supera 1, la variación por la semilla "
            "de entrenamiento pesa tanto como la del muestreo de imágenes.")


def _above_final_sentence(table: pd.DataFrame, final: str, split: str = "ood") -> str | None:
    """Grupos con más valor comercial puntual que el final en ``split`` (solo se reporta)."""
    sub = table[(table["kind"] == "metric") & (table["split"] == split)
                & (table["metric"] == "value_comercial")].set_index("group")
    if final not in sub.index:
        return None
    fin = sub.loc[final]
    above = sub[(sub["estimate"] > fin["estimate"]) & (sub.index != final)]
    if above.empty:
        return None
    items = []
    for group, r in above.iterrows():
        overlap = r["ci_lo"] <= fin["ci_hi"] and fin["ci_lo"] <= r["ci_hi"]
        items.append(f"{iteration_id(group)} ({_q(r['estimate'])}, IC95 "
                     f"{'solapado' if overlap else 'sin solaparse'} con el del final)")
    where = "OOD" if split == "ood" else "test"
    listed = items[0] if len(items) == 1 else f"{', '.join(items[:-1])} y {items[-1]}"
    return (f"- En {where}, {listed} {'supera' if len(items) == 1 else 'superan'} en valor "
            f"comercial puntual al modelo final ({_q(fin['estimate'])}). El final se eligió por "
            f"valor en validación antes del congelamiento y no se cambia con estos números.")


def conclusions(table: pd.DataFrame, final: str, refs: Sequence[str]) -> list[str]:
    lines = []
    for split in ("ood", "test"):
        for ref in refs:
            sentence = _overlap_sentence(table, final, ref, split)
            if sentence:
                lines.append(sentence)
    extra = [_above_final_sentence(table, final), _seeds_sentence(table)]
    return lines + [s for s in extra if s]


NOTES = [
    "- Bootstrap de percentiles sobre imágenes (remuestreo con reemplazo), con la misma semilla y "
    "el mismo número de réplicas que `cupo.evaluate`; la estimación puntual es la de la muestra "
    "completa.",
    "- Umbrales (t_L, t_H) y temperatura congelados de cada run, elegidos en validación; aquí no "
    "se elige ni se ajusta nada con test u OOD.",
    "- F1 macro y ocupado→libre con umbral 0.5; el valor, en el punto de operación de cada "
    "escenario. Ocupado→libre = P(mostrado libre | ocupado), el error más caro.",
    "- Con varias semillas: media ± desviación estándar entre semillas, y el IC95 es el promedio "
    "de los IC95 bootstrap de cada semilla (igual que `iteraciones.md`).",
    "- Δ pareado: en cada remuestra se usan las mismas imágenes (emparejadas por ruta) para los dos "
    "modelos. \"Final > ref.\" es la proporción de remuestras en que el final da más valor; no es "
    "un p-valor. Compara un solo run de cada modelo, así que no incluye la variación por la semilla "
    "de entrenamiento: una diferencia pequeña (como final − B2 en test) puede depender de la semilla.",
    "- Las imágenes de un mismo frame y día están correlacionadas (misma cámara, luz y clima), y "
    "el bootstrap por imagen las trata como independientes, así que los IC95 probablemente son "
    "más angostos de lo real, sobre todo en OOD.",
]


def render_markdown(table: pd.DataFrame, n_images: Mapping[str, int], final: str,
                    refs: Sequence[str], iterations_csv: Path) -> str:
    metric_rows = table[table["kind"] == "metric"]
    n_boot, seed = int(metric_rows["n_boot"].iloc[0]), int(metric_rows["seed"].iloc[0])
    lines = [
        "# Incertidumbre en test y OOD",
        "",
        f"Generada con `scripts/uncertainty.py` a partir de las predicciones congeladas de cada run. "
        f"IC95 por bootstrap ({n_boot:,} réplicas, semilla {seed}).",
        "",
    ]
    for split in SPLITS:
        lines += [f"## {SPLIT_NAMES[split]} (n = {n_images[split]:,})", "",
                  *split_table(table, split, final), ""]
    lines += ["## Diferencia pareada: modelo final frente a los baselines", "",
              *delta_table(table), "",
              "## Verificación", "", *verification_lines(table, iterations_csv), "",
              "## Conclusiones", "", *conclusions(table, final, refs), "",
              "## Notas", "", *NOTES, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def default_final() -> str:
    """``name`` del modelo servido (``models/best.json``) o, si no está, ``mnv3_ft_cost``."""
    card = paths.models_dir() / "best.json"
    if card.is_file():
        return str(load_json(card).get("name") or DEFAULT_FINAL)
    return DEFAULT_FINAL


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IC95 por bootstrap de métricas y valor en test y OOD (solo reporte, "
                    "con umbrales y temperatura congelados).")
    parser.add_argument("--allow-test", action="store_true",
                        help="obligatorio: lee predicciones de test y OOD (solo tras el congelamiento)")
    parser.add_argument("--runs-dir", default=None, help="carpeta de runs (por defecto runs/)")
    parser.add_argument("--out", default=None, help="carpeta de salida (por defecto results/)")
    parser.add_argument("--n-boot", type=int, default=None,
                        help=f"réplicas bootstrap (por defecto eval.n_boot de cada run o {DEFAULT_N_BOOT})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"semilla del bootstrap (por defecto {DEFAULT_SEED}, la de cupo.evaluate)")
    parser.add_argument("--final", default=None,
                        help="name del modelo final (por defecto el de models/best.json o mnv3_ft_cost)")
    parser.add_argument("--refs", nargs="+", default=list(DEFAULT_REFS),
                        help="name de las referencias para la diferencia pareada")
    parser.add_argument("--iterations", default=None,
                        help="iteraciones.csv para verificar (por defecto results/iteraciones.csv)")
    args = parser.parse_args(argv)
    if not args.allow_test:
        parser.error("Este reporte lee test y OOD: agrega --allow-test (solo tras el congelamiento).")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    # Las conclusiones usan símbolos (→, Δ) que algunas consolas no pueden mostrar.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args(argv)
    t0 = time.perf_counter()
    runs_dir = resolve_dir(args.runs_dir, paths.runs_dir())
    out_dir = resolve_dir(args.out, paths.results_dir())
    iterations_csv = resolve_dir(args.iterations, paths.results_dir() / "iteraciones.csv")
    final = args.final or default_final()
    refs = [r for r in args.refs if r != final]

    groups = load_groups(runs_dir, args.n_boot)
    if not groups:
        print(f"No hay runs con test_preds.npz y ood_preds.npz en {runs_dir}", file=sys.stderr)
        return 1
    n_images = check_same_images(groups)
    print(f"{sum(len(r) for r in groups.values())} runs en {len(groups)} grupos; mismas imágenes "
          f"en todos (test n={n_images['test']:,}, OOD n={n_images['ood']:,})")

    scenarios = load_scenarios("comercial")
    table = build_table(groups, final, refs, scenarios, args.seed, iterations_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{OUT_NAME}.csv", index=False)
    md = render_markdown(table, n_images, final, refs, iterations_csv)
    (out_dir / f"{OUT_NAME}.md").write_text(md, encoding="utf-8")

    for line in verification_lines(table, iterations_csv) + conclusions(table, final, refs):
        print(line)
    print(f"Escrito {out_dir / OUT_NAME}.csv y .md en {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
