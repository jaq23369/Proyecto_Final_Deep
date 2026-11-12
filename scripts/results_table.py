"""Tabla de iteraciones: junta todos los runs de ``runs/`` en ``results/``.

Uso:
    python scripts/results_table.py [--runs-dir runs] [--include-smoke] [--out results]

Escribe:
- ``iteraciones.md`` y ``iteraciones.csv``: una fila por grupo de runs (``name``);
- ``iteraciones_resumen.md``: una fila por iteración (B0, B1, ..., K6), compacta para el informe;
- ``ranking_sensibilidad_<escenario>.csv``: ¿cambia el modelo ganador al mover cada supuesto?

Fuente de verdad de val (no se usa ``metrics_val_eval.json``):
- ``summary.json`` para F1, tasa ocupado→libre, valor e IC95 de cada escenario;
- si existe ``calibration.json``, el run usa su temperatura y sus umbrales (igual que
  ``runs.load_run``), y el valor y el IC95 de cada escenario salen de ese archivo.
Si falta el valor guardado, se usa el recalculado (con aviso).

Con ``val_preds.npz`` (logits sin calibrar) y el punto de operación del run se recalculan con
``compute_value``: horas UVG, ``operadores_extra``, costo de inferencia y el valor sin franja
amarilla con el mejor umbral único ``t ∈ [0, 1]`` (elegido en val). El valor recalculado en el
punto de operación se compara con el guardado.

Test y OOD solo aparecen si el run ya tiene ``metrics_test.json`` / ``metrics_ood.json``
(se generan una única vez, en el congelamiento). Latencia y costo por predicción salen de
``benchmark.json``.

Agrupación y comparabilidad:
- Los runs se agrupan por ``name``: con varias semillas se reporta media ± desviación
  estándar; tiempo y costo de entrenamiento se reportan también en total.
- Los runs con un subconjunto de datos (``subset`` < 1, por ejemplo las pruebas de humo)
  forman un grupo aparte, ``"<name> (subset <f>)"``. No compiten con los runs completos: el
  Δ se mide solo contra filas con el mismo ``subset`` y el mejor run, los representantes y
  el ranking usan solo runs completos cuando los hay.
- B0 (status quo, sin sistema) es una fila sintética con valor 0, válida como referencia
  para cualquier ``subset``. El Δ de cada fila se mide contra la mejor fila comparable de la
  iteración anterior presente, en el orden B0, B1, B2, K1, ..., K6.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cupo import paths  # noqa: E402
from cupo.evaluate import probs_from_logits  # noqa: E402
from cupo.metrics import three_state_rates  # noqa: E402
from cupo.runs import load_json, load_npz, load_yaml, operating_point  # noqa: E402
from cupo.thresholds import curve_single_threshold  # noqa: E402
from cupo.value_model import compute_value, load_scenario, sensitivity, status_quo  # noqa: E402

SCENARIOS: tuple[str, ...] = ("comercial", "uvg")
PRIMARY = "comercial"
ITERATION_ORDER: tuple[str, ...] = ("B0", "B1", "B2", "K1", "K2", "K3", "K4", "K5", "K6")
SMOKE_DIRNAME = "_smoke"
EXTRA_SPLITS: tuple[str, ...] = ("test", "ood")
VALUE_RTOL = 1e-6  # tolerancia para comparar el valor recalculado con el guardado
B0_NAME = "b0_status_quo"
B0_NOTES = "Sin sistema: el conductor busca como hoy (T_sys = t_base); ahorro 0 y sin costos"
SENSITIVITY_FACTORS: tuple[float, ...] = (0.5, 0.8, 1.2, 1.5)
# Clave de ``benchmark.json`` (``cupo.benchmark``): ONNX Runtime en CPU, 1 hilo, batch 1.
LATENCY_KEY = "cpu_1_thread.per_slot.p50_ms"
BENCH_COST_KEY = "costo_por_1000_pred_Q"

UNITS = {"comercial": "Q/mes", "uvg": "Q/semestre"}
PERIOD = {"comercial": "mes", "uvg": "semestre"}

# Nombre en español (con unidad) de cada supuesto del modelo de valor.
PARAM_LABELS: dict[str, str] = {
    "espacios": "Espacios del parqueo (N)",
    "ocupacion": "Ocupación en hora pico (π)",
    "llegadas_dia": "Llegadas por día",
    "dias_mes": "Días de operación al mes",
    "dias_semestre": "Días de operación por semestre",
    "meses_semestre": "Meses por semestre",
    "horas_operacion_dia": "Horas de operación al día (h)",
    "refresh_s": "Refresco de la cámara (s)",
    "t_base_min": "Búsqueda sin sistema, t_base (min)",
    "t_acceso_min": "Acceso y estacionarse (min)",
    "t_recorrido_total_min": "Recorrido total de pasillos, L (min)",
    "t_redireccion_min": "Redirección tras un error (min)",
    "abandono_base": "Abandono sin sistema (prob.)",
    "ingreso_por_visita_Q": "Ingreso por visita (Q)",
    "valor_tiempo_Q_h": "Valor del tiempo (Q/h)",
    "valor_tiempo_estudiante_Q_h": "Valor del tiempo, estudiante (Q/h)",
    "valor_tiempo_personal_Q_h": "Valor del tiempo, personal (Q/h)",
    "frac_personal": "Fracción de llegadas de personal",
    "segundos_por_revision": "Revisión de un espacio amarillo (s)",
    "revisiones_por_hora_amber": "Revisiones por hora de cada amarillo",
    "fraccion_tiempo_operador": "Fracción del tiempo del encargado",
    "operadores_incluidos": "Encargados ya incluidos",
    "salario_operador_Q_h": "Salario del operador (Q/h)",
    "precision_verificacion": "Precisión de la verificación (prob.)",
    "n_camaras": "Cámaras",
    "costo_camara_Q": "Costo por cámara (Q)",
    "costo_servidor_mes_Q": "Servidor (Q/mes)",
    "meses_amortizacion": "Meses de amortización",
    "costo_por_1000_pred_Q": "Inferencia (Q por 1,000 pred.)",
    "costo_sensor_instalado_Q": "Sensor instalado (Q por espacio)",
    "mantenimiento_sensores_mes_Q": "Mantenimiento de sensores (Q/mes)",
}


def param_label(name: str) -> str:
    return PARAM_LABELS.get(name, name)


# ---------------------------------------------------------------------------
# Descubrimiento y carga de runs
# ---------------------------------------------------------------------------

def _children_with_summary(root: Path) -> list[Path]:
    """Subcarpetas de ``root`` con ``summary.json`` (las incompletas se ignoran con aviso)."""
    out = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if child.name.startswith("_"):
            continue
        if (child / "summary.json").exists():
            out.append(child)
        else:
            print(f"Aviso: {child.name} no tiene summary.json (¿sigue entrenando?); se omite")
    return out


def find_run_dirs(runs_dir: Path, include_smoke: bool = False) -> list[Path]:
    """Runs completos bajo ``runs_dir``; ``_smoke/`` solo con ``include_smoke``."""
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de runs: {runs_dir}")
    dirs = _children_with_summary(runs_dir)
    smoke = runs_dir / SMOKE_DIRNAME
    if include_smoke and smoke.is_dir():
        dirs += _children_with_summary(smoke)
    return dirs


def resolve_dir(value: str | None, default: Path) -> Path:
    """Ruta de la CLI: relativa al directorio actual o, si no existe, al repo."""
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (paths.repo_root() / path).resolve()


def iteration_id(name: str) -> str:
    """'k3_mnv3_ft' → 'K3'; si el nombre no sigue la convención, el propio nombre."""
    match = re.match(r"^([bBkK]\d+)(?:_|$)", name)
    return match.group(1).upper() if match else name


def iteration_rank(iteration: str) -> int:
    return ITERATION_ORDER.index(iteration) if iteration in ITERATION_ORDER else len(ITERATION_ORDER)


def _num(value: Any) -> float:
    """Número o NaN (None, texto o valores no finitos → NaN)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def subset_of(value: Any) -> float:
    """Fracción de datos del run; sin dato se asume el conjunto completo (1.0)."""
    subset = _num(value)
    return 1.0 if math.isnan(subset) or subset >= 1 else subset


@dataclass
class RunRecord:
    """Un run listo para la tabla y las figuras (sin cargar el modelo)."""

    run_dir: Path
    summary: dict[str, Any]
    config: dict[str, Any]
    temperature: float
    thresholds: dict[str, dict[str, float]]
    calibrated: bool
    calibration: dict[str, Any] | None
    val: dict[str, np.ndarray] | None

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    @property
    def name(self) -> str:
        return str(self.summary.get("name") or self.config.get("name") or self.run_id)

    @property
    def iteration(self) -> str:
        return iteration_id(self.name)

    @property
    def subset(self) -> float:
        return subset_of(self.summary.get("subset"))

    @property
    def is_full(self) -> bool:
        return self.subset >= 1

    @property
    def group(self) -> str:
        """Clave de agrupación: ``name`` y, si el run usó un subconjunto, la fracción."""
        return self.name if self.is_full else f"{self.name} (subset {self.subset:g})"

    def val_p(self) -> np.ndarray | None:
        """P(ocupado) en val con la temperatura del run (1 si no está calibrado)."""
        if self.val is None:
            return None
        if "logits" in self.val:
            return probs_from_logits(self.val["logits"], self.temperature)
        return np.asarray(self.val["p"], dtype=float)


def _optional_json(path: Path) -> dict[str, Any] | None:
    """JSON si existe y se puede leer; si está corrupto, avisa y devuelve None."""
    if not path.exists():
        return None
    try:
        data = load_json(path)
    except (OSError, ValueError) as exc:
        print(f"Aviso: no se pudo leer {path.parent.name}/{path.name} ({exc}); se ignora")
        return None
    return data if isinstance(data, dict) else None


def _optional_npz(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        data = load_npz(path)
    except (OSError, ValueError) as exc:
        print(f"Aviso: no se pudo leer {path.parent.name}/{path.name} ({exc}); se ignora")
        return None
    return data if {"y"} <= set(data) and ({"logits"} <= set(data) or {"p"} <= set(data)) else None


def load_record(run_dir: Path) -> RunRecord:
    """Lee summary, config, punto de operación y ``val_preds.npz`` de un run."""
    summary = load_json(run_dir / "summary.json")
    if not isinstance(summary, dict):
        raise ValueError("summary.json no es un objeto JSON")
    temperature, thresholds, calibrated = operating_point(run_dir)
    cfg_path = run_dir / "config.yaml"
    return RunRecord(
        run_dir=run_dir,
        summary=summary,
        config=(load_yaml(cfg_path) or {}) if cfg_path.exists() else {},
        temperature=temperature,
        thresholds=thresholds if isinstance(thresholds, dict) else {},
        calibrated=calibrated,
        calibration=_optional_json(run_dir / "calibration.json"),
        val=_optional_npz(run_dir / "val_preds.npz"),
    )


def load_records(runs_dir: Path, include_smoke: bool = False) -> list[RunRecord]:
    """Carga todos los runs; los que no se pueden leer se saltan con aviso."""
    records = []
    for run_dir in find_run_dirs(runs_dir, include_smoke):
        try:
            records.append(load_record(run_dir))
        except Exception as exc:  # noqa: BLE001 - un run dañado no debe tumbar la tabla
            print(f"Aviso: se omite {run_dir.name}: {type(exc).__name__}: {exc}")
    return records


def load_scenarios() -> dict[str, dict[str, Any]]:
    return {name: load_scenario(name) for name in SCENARIOS}


# ---------------------------------------------------------------------------
# Selección: runs comparables, representantes y mejor run
# ---------------------------------------------------------------------------

def _ci(value: Any) -> tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        return _num(value[0]), _num(value[1])
    return float("nan"), float("nan")


def stored_value(rec: RunRecord, scenario: str) -> tuple[float, float, float]:
    """(valor, IC95 inferior, IC95 superior) guardados en el run (summary o calibration)."""
    if rec.calibrated:
        value_val = (rec.calibration or {}).get("value_val") or {}
        entry = value_val.get(scenario) or {}
        return (_num(entry.get("objective_value")), *_ci(entry.get("ci95")))
    return (_num(rec.summary.get(f"val_value_{scenario}")),
            *_ci(rec.summary.get(f"val_value_{scenario}_ci95")))


def operating_rates(rec: RunRecord, scenario: str, p: np.ndarray | None = None) -> dict[str, Any] | None:
    """Tasas de tres estados en val con los umbrales del run para ``scenario``."""
    thr = rec.thresholds.get(scenario)
    if p is None:
        p = rec.val_p()
    if thr is None or p is None or "t_L" not in thr or "t_H" not in thr:
        return None
    return three_state_rates(rec.val["y"], p, float(thr["t_L"]), float(thr["t_H"]))


def source_value(rec: RunRecord, scenario: str) -> tuple[float, float, float]:
    """Valor en val del run: el guardado o, si falta, el recalculado (sin IC)."""
    value, lo, hi = stored_value(rec, scenario)
    if not math.isnan(value):
        return value, lo, hi
    rates = operating_rates(rec, scenario)
    if rates is None:
        return value, lo, hi
    return float(compute_value(rates, load_scenario(scenario))["objective"]), float("nan"), float("nan")


def record_value(rec: RunRecord, scenario: str = PRIMARY) -> float:
    value = source_value(rec, scenario)[0]
    return -math.inf if math.isnan(value) else value


def comparable_records(records: Sequence[RunRecord], warn: bool = True) -> list[RunRecord]:
    """Runs completos (``subset`` = 1) si hay alguno; si no, todos (con aviso)."""
    full = [r for r in records if r.is_full]
    if full:
        return full
    if records and warn:
        print("Aviso: no hay runs con el conjunto completo (subset = 1); mejor run, "
              "representantes y ranking se calculan con subconjuntos y no son concluyentes")
    return list(records)


def representatives(records: Sequence[RunRecord], warn: bool = False) -> list[RunRecord]:
    """Un run por grupo comparable (el de mayor valor comercial en val), en orden de iteración."""
    best: dict[str, RunRecord] = {}
    for rec in comparable_records(records, warn):
        if rec.group not in best or record_value(rec) > record_value(best[rec.group]):
            best[rec.group] = rec
    return sorted(best.values(), key=lambda r: (iteration_rank(r.iteration), r.iteration, r.group))


def best_record(records: Sequence[RunRecord], warn: bool = False) -> RunRecord | None:
    """Run comparable de mayor valor comercial en val (con predicciones de val)."""
    usable = [r for r in comparable_records(records, warn) if r.val is not None]
    return max(usable, key=record_value) if usable else None


# ---------------------------------------------------------------------------
# Métricas de cada run
# ---------------------------------------------------------------------------

def best_single_threshold(y: np.ndarray, p: np.ndarray, cfg: Mapping) -> tuple[float, float]:
    """(t, valor) del mejor umbral único t ∈ [0, 1] (sin franja amarilla) en val.

    En empate gana el t más cercano a 0.5.
    """
    curve = curve_single_threshold(y, p, cfg)
    values = curve["value"].to_numpy(dtype=float)
    best = np.nanmax(values)
    tied = np.flatnonzero(values >= best - 1e-9 * max(1.0, abs(best)))
    t = curve["t"].to_numpy()[tied]
    k = int(tied[np.argmin(np.abs(t - 0.5))])
    return float(curve["t"].iloc[k]), float(values[k])


def recomputed_values(rec: RunRecord, scenarios: Mapping[str, Mapping]) -> dict[str, dict[str, Any]]:
    """Componentes recalculados desde ``val_preds.npz``: punto de operación y mejor umbral único."""
    p = rec.val_p()
    if p is None:
        return {}
    y = rec.val["y"]
    out: dict[str, dict[str, Any]] = {}
    for name, cfg in scenarios.items():
        t1, v1 = best_single_threshold(y, p, cfg)
        entry: dict[str, Any] = {"single_t": t1, "single_value": v1}
        rates = operating_rates(rec, name, p)
        if rates is not None:
            entry["rates"] = rates
            entry["components"] = compute_value(rates, cfg)
        out[name] = entry
    return out


def values_match(rec: RunRecord, scenario: str, stored: float, recomputed: float) -> bool:
    """¿Coincide el valor recalculado con el guardado? Avisa si no.

    Una diferencia suele indicar que el run se guardó con una versión anterior del modelo
    de valor (``configs/value`` o ``value_model.py``); la tabla conserva el valor guardado.
    """
    if math.isnan(stored) or math.isnan(recomputed):
        return True
    if math.isclose(stored, recomputed, rel_tol=VALUE_RTOL, abs_tol=1e-6):
        return True
    print(f"Aviso: {rec.run_id} [{scenario}] valor guardado {stored:,.2f} y recalculado "
          f"{recomputed:,.2f} no coinciden")
    return False


def split_metrics(run_dir: Path, split: str) -> dict[str, float]:
    """F1, tasa ocupado→libre y valor comercial de ``metrics_{split}.json`` (si existe)."""
    data = _optional_json(run_dir / f"metrics_{split}.json")
    if data is None:
        return {}
    binary = data.get("binary") or {}
    return {
        f"{split}_f1_macro": _num(binary.get("f1_macro")),
        f"{split}_rate_busy_as_free": _num(binary.get("rate_busy_as_free")),
        f"{split}_value_comercial": _num((data.get("value") or {}).get(PRIMARY)),
    }


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, val in obj.items():
            out.update(_flatten(val, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: obj}


_SLOT_HINTS = ("slot", "espacio", "batch1", "batch_1", "bs1", "per_patch", "patch")


def latency_ms(bench: Mapping[str, Any] | None) -> float:
    """Latencia p50 por espacio en CPU (ms) de ``benchmark.json``; NaN si no hay.

    Usa ``LATENCY_KEY``; si el archivo tiene otro esquema, toma la clave numérica con
    "p50" que más se parezca a CPU y un solo espacio (batch 1).
    """
    if not bench:
        return float("nan")
    flat = {k.lower(): v for k, v in _flatten(bench).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if LATENCY_KEY in flat:
        return _num(flat[LATENCY_KEY])
    candidates = [k for k in flat if "p50" in k]
    if not candidates:
        return float("nan")

    def score(key: str) -> tuple[bool, bool, bool]:
        return ("cpu" in key or "ort" in key, any(h in key for h in _SLOT_HINTS), "ms" in key)

    best = max(candidates, key=score)
    value = _num(flat[best])
    in_seconds = "ms" not in best and (best.endswith("_s") or "seconds" in best)
    return value * 1000.0 if in_seconds else value


def inference_cost(comp: Mapping[str, Any] | None, bench: Mapping[str, Any] | None) -> tuple[float, str]:
    """(costo de inferencia en Q/mes del escenario comercial, fuente del costo por predicción)."""
    if not comp:
        return float("nan"), ""
    bench_cost = _num((bench or {}).get(BENCH_COST_KEY))
    if not math.isnan(bench_cost):
        return float(comp["pred_mes"]) / 1000.0 * bench_cost, "benchmark.json"
    return float(comp["costo_inf_mes"]), "costs.yaml"


def _base_row(rec: RunRecord, bench: Mapping[str, Any] | None) -> dict[str, Any]:
    s = rec.summary
    seed = _num(s.get("seed"))
    return {
        "iteration": rec.iteration,
        "group": rec.group,
        "name": rec.name,
        "run_id": rec.run_id,
        "model": s.get("model", ""),
        "notes": s.get("notes", ""),
        "seed": None if math.isnan(seed) else int(seed),
        "subset": rec.subset,
        "calibrated": rec.calibrated,
        "temperature": rec.temperature,
        "params_total": _num(s.get("params_total")),
        "params_trainable": _num(s.get("params_trainable")),
        "size_mb": _num(s.get("size_mb")),
        "train_acc": _num(s.get("train_acc")),
        "val_f1_macro": _num(s.get("val_f1_macro")),
        "val_rate_busy_as_free": _num(s.get("val_rate_busy_as_free")),
        "latency_ms": latency_ms(bench),
        "cost_per_1000_pred_q": _num((bench or {}).get(BENCH_COST_KEY)),
        "train_time_s": _num(s.get("train_time_s")),
        "train_cost_q": _num(s.get("train_cost_q")),
        "train_cost_usd": _num(s.get("train_cost_usd")),
    }


def _scenario_columns(rec: RunRecord, name: str, entry: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Columnas de un escenario y si el valor guardado coincide con el recalculado."""
    comp = entry.get("components")
    stored, lo, hi = stored_value(rec, name)
    recomputed = _num(comp["objective"]) if comp else float("nan")
    value = stored
    if math.isnan(stored) and not math.isnan(recomputed):
        print(f"Aviso: {rec.run_id} [{name}] no tiene valor guardado; se usa el recalculado")
        value = recomputed
    thr = rec.thresholds.get(name) or {}
    cols = {
        f"val_value_{name}": value,
        f"val_value_{name}_ci_lo": lo,
        f"val_value_{name}_ci_hi": hi,
        f"t_L_{name}": _num(thr.get("t_L")),
        f"t_H_{name}": _num(thr.get("t_H")),
        f"val_verify_rate_{name}": _num((entry.get("rates") or {}).get("verify_rate")),
        f"operadores_extra_{name}": _num(comp["operadores_extra"]) if comp else float("nan"),
        f"val_value_{name}_1thr": _num(entry.get("single_value")),
        f"t_1thr_{name}": _num(entry.get("single_t")),
        f"val_value_{name}_recomputed": recomputed,
    }
    return cols, values_match(rec, name, stored, recomputed)


def run_row(rec: RunRecord, scenarios: Mapping[str, Mapping]) -> dict[str, Any]:
    """Fila plana de un run (antes de agrupar)."""
    bench = _optional_json(rec.run_dir / "benchmark.json")
    row = _base_row(rec, bench)
    rec_vals = recomputed_values(rec, scenarios)
    row["value_mismatch"] = False
    for name in scenarios:
        cols, ok = _scenario_columns(rec, name, rec_vals.get(name, {}))
        row.update(cols)
        row["value_mismatch"] |= not ok
    row["inference_cost_month_q"], row["inference_cost_source"] = inference_cost(
        rec_vals.get(PRIMARY, {}).get("components"), bench)
    uvg_comp = rec_vals.get("uvg", {}).get("components")
    hours = _num(rec.summary.get("val_hours_uvg"))
    if (rec.calibrated or math.isnan(hours)) and uvg_comp:
        hours = _num(uvg_comp["horas_ahorradas_semestre"])
    row["val_hours_uvg"] = hours
    for split in EXTRA_SPLITS:
        row.update(split_metrics(rec.run_dir, split))
    return row


# ---------------------------------------------------------------------------
# Agrupación, B0 y Δ
# ---------------------------------------------------------------------------

TEXT_COLUMNS = ("iteration", "group", "name", "model", "notes")
NON_NUMERIC = (*TEXT_COLUMNS, "run_id", "seed", "calibrated", "value_mismatch",
               "inference_cost_source")
TOTAL_COLUMNS = ("train_time_s", "train_cost_q", "train_cost_usd")


def _calibrated_label(values: Iterable[bool]) -> str:
    vals = set(bool(v) for v in values)
    return "mixto" if len(vals) > 1 else ("sí" if vals == {True} else "no")


def _seeds_text(seeds: Iterable[Any]) -> str:
    return ", ".join("–" if s is None or _isnan(s) else str(int(s)) for s in seeds)


def _aggregate_group(grp: pd.DataFrame, numeric: Sequence[str]) -> dict[str, Any]:
    first = grp.iloc[0]
    out: dict[str, Any] = {c: first[c] for c in TEXT_COLUMNS}
    out["n_runs"] = len(grp)
    out["seeds"] = _seeds_text(grp["seed"].tolist())
    out["run_ids"] = "; ".join(grp["run_id"].tolist())
    out["calibrated"] = _calibrated_label(grp["calibrated"])
    out["value_mismatch"] = bool(grp["value_mismatch"].any())
    out["inference_cost_source"] = "; ".join(sorted(set(grp["inference_cost_source"]) - {""}))
    for col in numeric:
        vals = grp[col].astype(float)
        out[col] = vals.mean() if vals.notna().any() else float("nan")
        out[f"{col}_std"] = vals.std(ddof=1) if vals.notna().sum() > 1 else float("nan")
    for col in TOTAL_COLUMNS:
        vals = grp[col].astype(float)
        out[f"{col}_total"] = vals.sum() if vals.notna().any() else float("nan")
    return out


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Una fila por grupo: media, ``<col>_std`` (ddof=1) y totales de entrenamiento."""
    numeric = [c for c in rows.columns
               if c not in NON_NUMERIC and pd.api.types.is_numeric_dtype(rows[c])]
    return pd.DataFrame([_aggregate_group(grp, numeric)
                         for _, grp in rows.groupby("group", sort=False)])


def b0_row(scenarios: Mapping[str, Mapping]) -> dict[str, Any]:
    """Fila sintética del status quo (valor 0 en ambos escenarios)."""
    row: dict[str, Any] = {"iteration": "B0", "group": B0_NAME, "name": B0_NAME,
                           "model": "sin sistema", "notes": B0_NOTES, "n_runs": 0, "seeds": "",
                           "run_ids": "", "calibrated": "", "subset": 1.0}
    for name, cfg in scenarios.items():
        base = status_quo(cfg)
        row[f"val_value_{name}"] = float(base["objective"])
        row[f"val_value_{name}_1thr"] = float(base["objective"])
        row[f"operadores_extra_{name}"] = 0.0
    row["val_hours_uvg"] = (float(status_quo(scenarios["uvg"])["horas_ahorradas_semestre"])
                            if "uvg" in scenarios else 0.0)
    return row


def sort_table(table: pd.DataFrame) -> pd.DataFrame:
    keys = table["iteration"].map(iteration_rank)
    return (table.assign(_rank=keys).sort_values(["_rank", "iteration", "group"])
            .drop(columns="_rank").reset_index(drop=True))


def _best_row_by_iteration(table: pd.DataFrame, col: str, subset: float) -> dict[str, int]:
    """Índice de la mejor fila de cada iteración entre las de ``subset`` (B0 vale para todos)."""
    pool = table[(table["subset"].astype(float) == subset) | (table["iteration"] == "B0")]
    best: dict[str, int] = {}
    for it, grp in pool.groupby("iteration", sort=False):
        vals = pd.to_numeric(grp[col], errors="coerce").dropna()
        if len(vals):
            best[str(it)] = int(vals.idxmax())
    return best


def _delta_for_row(table: pd.DataFrame, row: pd.Series, col: str) -> tuple[float, str]:
    """Δ frente a la mejor fila comparable de la iteración anterior presente."""
    if row["iteration"] == "B0":
        return float("nan"), ""
    subset = float(row["subset"])
    best = _best_row_by_iteration(table, col, subset)
    order = [it for it in dict.fromkeys(table["iteration"]) if it in best]
    if row["iteration"] not in order or order.index(row["iteration"]) == 0:
        return float("nan"), ""
    prev = order[order.index(row["iteration"]) - 1]
    ref = table.loc[best[prev]]
    pool = table[(table["iteration"] == prev) & (table["subset"].astype(float) == subset)]
    ref_label = prev if prev == "B0" or len(pool) <= 1 else f"{prev} ({ref['group']})"
    return float(row[col]) - float(ref[col]), ref_label


def add_deltas(table: pd.DataFrame, scenarios: Iterable[str] = SCENARIOS) -> pd.DataFrame:
    """Δ valor de cada fila frente a la mejor fila comparable de la iteración anterior.

    "Comparable" = mismo ``subset``; B0 sirve de referencia para cualquiera. ``table`` debe
    venir ordenada (``sort_table``); ``delta_ref_<escenario>`` indica la referencia.
    """
    table = table.copy()
    for scen in scenarios:
        col = f"val_value_{scen}"
        pairs = [_delta_for_row(table, row, col) for _, row in table.iterrows()]
        table[f"delta_value_{scen}"] = [d for d, _ in pairs]
        table[f"delta_ref_{scen}"] = [r for _, r in pairs]
    return table


def build_table(records: Sequence[RunRecord], scenarios: Mapping[str, Mapping]) -> pd.DataFrame:
    """Tabla de iteraciones completa (B0 + una fila por grupo + Δ)."""
    rows = pd.DataFrame([run_row(r, scenarios) for r in records])
    table = aggregate(rows) if len(rows) else pd.DataFrame()
    table = pd.concat([pd.DataFrame([b0_row(scenarios)]), table], ignore_index=True)
    table["subset"] = table["subset"].astype(float).fillna(1.0)
    return add_deltas(sort_table(table))


# ---------------------------------------------------------------------------
# ¿Cambia el ranking de modelos si los supuestos del modelo de valor varían?
# ---------------------------------------------------------------------------

def ranking_sensitivity(reps: Sequence[RunRecord], name: str, cfg: Mapping,
                        factors: Sequence[float] = SENSITIVITY_FACTORS) -> pd.DataFrame:
    """Ganador por (parámetro, factor) con las tasas fijas de cada representante.

    Cada modelo conserva su punto de operación (umbrales elegidos en val); solo cambia el
    supuesto. Columnas: param, factor, param_value, clipped, una columna de valor por grupo,
    winner, runner_up, margin, base_winner, winner_changed, order_changed.
    """
    tables = {}
    for rec in reps:
        rates = operating_rates(rec, name)
        if rates is not None:
            tables[rec.group] = sensitivity(rates, cfg, factors=factors).set_index(["param", "factor"])
    if len(tables) < 2:
        return pd.DataFrame()
    groups = list(tables)
    first = tables[groups[0]]
    out = first[["param_value", "clipped"]].copy()
    for g in groups:
        out[g] = tables[g]["objective"]
    base = pd.Series({g: float(tables[g]["base_objective"].iloc[0]) for g in groups})
    base_order = list(base.sort_values(ascending=False).index)
    vals = out[groups].astype(float)
    order = vals.apply(lambda r: list(r.sort_values(ascending=False).index), axis=1)
    out["winner"] = order.map(lambda o: o[0])
    out["runner_up"] = order.map(lambda o: o[1])
    out["margin"] = vals.apply(lambda r: float(np.diff(np.sort(r.to_numpy())[-2:])[0]), axis=1)
    out["base_winner"] = base_order[0]
    out["winner_changed"] = out["winner"] != base_order[0]
    out["order_changed"] = order.map(lambda o: o != base_order)
    return out.reset_index()


def ranking_summary(rank: pd.DataFrame, name: str, max_items: int = 8) -> list[str]:
    """Líneas de resumen para el .md."""
    label = name.capitalize() if name != "uvg" else "UVG"
    if rank.empty:
        return [f"- **{label}**: se necesitan al menos 2 modelos comparables para el ranking."]
    n, changed = len(rank), rank[rank["winner_changed"]]
    n_order = int(rank["order_changed"].sum())
    head = (f"- **{label}**: en el caso base gana **{rank['base_winner'].iloc[0]}**. El ganador "
            f"cambia en {len(changed)} de {n} variaciones (cada supuesto ×0.5, ×0.8, ×1.2, ×1.5) "
            f"y el orden completo cambia en {n_order} de {n}")
    if changed.empty:
        return [head + "."]
    items = [f"{param_label(r['param'])} ×{r['factor']:g} → {r['winner']}"
             for _, r in changed.head(max_items).iterrows()]
    more = f" y {len(changed) - max_items} más" if len(changed) > max_items else ""
    return [head + ": " + "; ".join(items) + more + "."]


# ---------------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------------

def _isnan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt_num(x: Any, decimals: int = 0, pct: bool = False) -> str:
    if _isnan(x):
        return "–"
    val = float(x) * (100 if pct else 1)
    text = f"{val:,.{decimals}f}"
    return f"{text} %" if pct else text


def fmt_mean_std(row: Mapping[str, Any], col: str, decimals: int = 0, pct: bool = False) -> str:
    """'media ± desviación' si hay varias semillas; si no, solo el valor."""
    base = fmt_num(row.get(col), decimals, pct)
    std = row.get(f"{col}_std")
    if base == "–" or _isnan(std):
        return base
    return f"{base} ± {fmt_num(std, decimals, pct)}"


def fmt_ci(row: Mapping[str, Any], col: str) -> str:
    lo, hi = row.get(f"{col}_ci_lo"), row.get(f"{col}_ci_hi")
    if _isnan(row.get(col)) or _isnan(lo) or _isnan(hi):
        return ""
    return f" [{fmt_num(lo)}; {fmt_num(hi)}]"


def fmt_delta(x: Any) -> str:
    return "–" if _isnan(x) else f"{float(x):+,.0f}"


def fmt_params(x: Any) -> str:
    if _isnan(x):
        return "–"
    x = float(x)
    if x >= 1e6:
        return f"{x / 1e6:.2f} M"
    if x >= 1e3:
        return f"{x / 1e3:.1f} k"
    return f"{x:.0f}"


def fmt_time(seconds: Any) -> str:
    if _isnan(seconds):
        return "–"
    s = float(seconds)
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


def _md_escape(text: Any) -> str:
    return "" if _isnan(text) else str(text).replace("|", "\\|").replace("\n", " ")


def row_label(row: Mapping[str, Any]) -> str:
    flag = row.get("value_mismatch")
    mark = " †" if not _isnan(flag) and bool(flag) else ""
    return f"**{row['iteration']}** ({row['group']}){mark}"


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(_md_escape(c) for c in r) + " |" for r in rows]
    return lines


def _delta_text(row: Mapping[str, Any], scen: str) -> str:
    delta = fmt_delta(row.get(f"delta_value_{scen}"))
    ref = row.get(f"delta_ref_{scen}")
    return f"{delta} (vs. {ref})" if delta != "–" and ref else delta


def _single_text(row: Mapping[str, Any], scen: str) -> str:
    value = fmt_mean_std(row, f"val_value_{scen}_1thr")
    t = row.get(f"t_1thr_{scen}")
    return value if _isnan(t) else f"{value} (t = {fmt_mean_std(row, f't_1thr_{scen}', 2)})"


def main_table_md(table: pd.DataFrame) -> list[str]:
    headers = ["ID", "Modelo", "Cambios", "Semillas", "Train acc", "Val F1 macro",
               "Val ocupado→libre", "Val valor comercial (Q/mes) [IC95]",
               "Δ valor vs. iteración anterior (Q/mes)",
               "Valor comercial sin franja, mejor umbral único (Q/mes)",
               "Operadores extra (comercial)", "Val horas UVG (h/semestre)",
               "Val valor UVG (Q/semestre)", "Valor UVG sin franja, mejor umbral único (Q/semestre)",
               "Operadores extra (UVG)"]
    rows = []
    for _, r in table.iterrows():
        rows.append([
            row_label(r), r["model"], r["notes"], r.get("seeds") or "–",
            fmt_mean_std(r, "train_acc", 4), fmt_mean_std(r, "val_f1_macro", 4),
            fmt_mean_std(r, "val_rate_busy_as_free", 2, pct=True),
            fmt_mean_std(r, "val_value_comercial") + fmt_ci(r, "val_value_comercial"),
            _delta_text(r, "comercial"), _single_text(r, "comercial"),
            fmt_mean_std(r, "operadores_extra_comercial", 2), fmt_mean_std(r, "val_hours_uvg"),
            fmt_mean_std(r, "val_value_uvg") + fmt_ci(r, "val_value_uvg"),
            _single_text(r, "uvg"), fmt_mean_std(r, "operadores_extra_uvg", 2),
        ])
    return _md_table(headers, rows)


def _thr_text(row: Mapping[str, Any], scen: str) -> str:
    t_L, t_H = row.get(f"t_L_{scen}"), row.get(f"t_H_{scen}")
    if _isnan(t_L) or _isnan(t_H):
        return "–"
    note = " (media)" if not _isnan(row.get(f"t_L_{scen}_std")) else ""
    return f"{t_L:.2f} / {t_H:.2f}{note}"


def _total_text(row: Mapping[str, Any], col: str, fmt) -> str:
    """Total del grupo; con varias semillas agrega la media por run."""
    total = fmt(row.get(f"{col}_total"))
    n = row.get("n_runs") or 0
    return f"{total} ({n} runs; {fmt(row.get(col))} c/u)" if n > 1 and total != "–" else total


def _inference_text(row: Mapping[str, Any]) -> str:
    value = fmt_mean_std(row, "inference_cost_month_q", 0)
    src = row.get("inference_cost_source")
    return f"{value} ({src})" if value != "–" and src else value


def ops_table_md(table: pd.DataFrame) -> list[str]:
    headers = ["ID", "Params", "Tamaño (MB)", "Latencia CPU p50 (ms/espacio)",
               "Costo de inferencia (Q/mes, comercial)", "Tiempo de entrenamiento (total)",
               "Costo de entrenamiento total (Q)", "Calibrado (T)", "t_L / t_H comercial",
               "Por verificar (val, comercial)", "t_L / t_H UVG"]
    rows = []
    for _, r in table.iterrows():
        if r["iteration"] == "B0":
            continue
        calib = r.get("calibrated", "")
        if calib in ("sí", "mixto"):
            calib = f"{calib} (T = {fmt_mean_std(r, 'temperature', 3)})"
        rows.append([
            row_label(r), fmt_params(r.get("params_total")), fmt_mean_std(r, "size_mb", 2),
            fmt_mean_std(r, "latency_ms", 2), _inference_text(r),
            _total_text(r, "train_time_s", fmt_time),
            _total_text(r, "train_cost_q", lambda x: fmt_num(x, 3)), calib,
            _thr_text(r, "comercial"), fmt_mean_std(r, "val_verify_rate_comercial", 2, pct=True),
            _thr_text(r, "uvg"),
        ])
    return _md_table(headers, rows)


def has_split(table: pd.DataFrame, split: str) -> bool:
    col = f"{split}_f1_macro"
    return col in table.columns and table[col].notna().any()


def eval_table_md(table: pd.DataFrame, splits: Sequence[str]) -> list[str]:
    names = {"test": "Test", "ood": "OOD (CNRPark A/B)"}
    headers = ["ID"]
    for sp in splits:
        headers += [f"{names[sp]} F1 macro", f"{names[sp]} ocupado→libre",
                    f"{names[sp]} valor comercial (Q/mes)"]
    rows = []
    for _, r in table.iterrows():
        if r["iteration"] == "B0":
            continue
        cells = [row_label(r)]
        for sp in splits:
            cells += [fmt_mean_std(r, f"{sp}_f1_macro", 4),
                      fmt_mean_std(r, f"{sp}_rate_busy_as_free", 2, pct=True),
                      fmt_mean_std(r, f"{sp}_value_comercial")]
        rows.append(cells)
    return _md_table(headers, rows)


NOTES = [
    "- **B0** es el status quo (sin sistema): fila sintética con valor 0 en ambos escenarios.",
    "- Con varias semillas cada celda es media ± desviación estándar; el IC95 es el promedio "
    "de los IC95 bootstrap de cada semilla. Tiempo y costo de entrenamiento se suman.",
    "- **Valor**: comercial = valor para el operador (Q/mes); UVG = valor neto del tiempo "
    "ahorrado (Q/semestre). Se usan los umbrales (t_L, t_H) óptimos en val; en runs "
    "calibrados, la temperatura y los umbrales de `calibration.json`.",
    "- **Δ valor**: diferencia frente a la mejor fila comparable (mismo `subset`; B0 vale para "
    "todas) de la iteración anterior presente, en el orden B0, B1, B2, K1…K6.",
    "- **Sin franja, mejor umbral único**: decisión de 2 estados (t_L = t_H = t) con el t ∈ [0, 1] "
    "que maximiza el valor en val. **Operadores extra**: personal adicional (tiempo completo "
    "equivalente) para revisar los espacios por verificar en el punto de operación.",
    "- **Costo de inferencia**: predicciones del mes del escenario comercial × costo por 1,000 "
    "predicciones (de `benchmark.json` si existe; si no, el de `configs/costs.yaml`).",
    "- **Train acc**: en los modelos de DL es la media de la época (con aumento y dropout); "
    "en B1 es la accuracy del modelo final sobre train.",
    "- Ocupado→libre: P(mostrado libre | ocupado) con umbral 0.5, el error más caro.",
]


def render_markdown(table: pd.DataFrame, runs_dir: Path, include_smoke: bool,
                    rankings: Mapping[str, pd.DataFrame]) -> str:
    n_runs = int(pd.to_numeric(table["n_runs"], errors="coerce").fillna(0).sum())
    lines = ["# Tabla de iteraciones", "",
             f"Generada con `scripts/results_table.py` a partir de {n_runs} run(s) en "
             f"`{runs_dir.name}/`" + (" (incluye pruebas de humo)" if include_smoke else "") + ".", ""]
    if (table["subset"].astype(float) < 1).any():
        lines += ["> **Atención:** hay runs entrenados y validados con un subconjunto de los datos "
                  "(`subset` < 1). No son comparables con los completos: su Δ solo se mide contra "
                  "el mismo `subset` y no entran al mejor run ni al ranking si hay runs completos.", ""]
    lines += ["## Validación", ""] + main_table_md(table) + [""]
    lines += ["## Costo y operación", ""] + ops_table_md(table) + [""]
    splits = [sp for sp in EXTRA_SPLITS if has_split(table, sp)]
    if splits:
        lines += ["## Test y fuera de dominio (evaluación única tras el congelamiento)", "",
                  "Umbrales y temperatura elegidos en val; nunca se re-optimizan en test ni en OOD.",
                  ""] + eval_table_md(table, splits) + [""]
    lines += ["## ¿Cambia el ranking con los supuestos?", "",
              "Cada modelo conserva sus tasas en val y su punto de operación; se mueve un supuesto "
              "a la vez. Detalle en `ranking_sensibilidad_<escenario>.csv`.", ""]
    has_full = ((table["subset"].astype(float) >= 1) & (table["iteration"] != "B0")).any()
    if not has_full:
        lines += ["> No hay runs con el conjunto completo: el ranking usa subconjuntos (pruebas de "
                  "humo) y no es concluyente.", ""]
    elif (table["subset"].astype(float) < 1).any():
        lines += ["Solo participan los runs con el conjunto completo (un representante por grupo).", ""]
    for name, rank in rankings.items():
        lines += ranking_summary(rank, name)
    lines += ["", "## Notas", "", *NOTES]
    if any(not _isnan(f) and bool(f) for f in table.get("value_mismatch", [])):
        lines.append("- **†** El valor guardado en el run no coincide con el recalculado con el "
                     "modelo de valor actual (el run se generó con una versión anterior de "
                     "`configs/value` o `value_model.py`). La tabla muestra el valor guardado; "
                     "el recalculado está en `val_value_*_recomputed` del CSV. Conviene volver "
                     "a generar el run para compararlo con los demás.")
    return "\n".join(lines) + "\n"


def summary_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Una fila por iteración: la mejor fila comparable (completa si hay) por valor comercial."""
    full = table[table["subset"].astype(float) >= 1]
    pool = full if (full["iteration"] != "B0").any() else table
    col = f"val_value_{PRIMARY}"
    idx = [int(pd.to_numeric(g[col], errors="coerce").fillna(-np.inf).idxmax())
           for _, g in pool.groupby("iteration", sort=False)]
    return sort_table(table.loc[idx])


def render_summary(table: pd.DataFrame) -> str:
    """``iteraciones_resumen.md``: compacto para §4.5 del informe."""
    rows = summary_rows(table)
    headers = ["Iteración", "Modelo", "Train acc", "Val F1 macro", "Val valor comercial (Q/mes)",
               "Test F1 macro", "Test valor comercial (Q/mes)", "Latencia CPU p50 (ms/espacio)",
               "Costo de entrenamiento total (Q)"]
    body = [[row_label(r), r["model"], fmt_mean_std(r, "train_acc", 4),
             fmt_mean_std(r, "val_f1_macro", 4), fmt_mean_std(r, f"val_value_{PRIMARY}"),
             fmt_mean_std(r, "test_f1_macro", 4), fmt_mean_std(r, "test_value_comercial"),
             fmt_mean_std(r, "latency_ms", 2), fmt_num(r.get("train_cost_q_total"), 3)]
            for _, r in rows.iterrows()]
    lines = ["# Resumen de iteraciones", "",
             "Una fila por iteración (si hay variantes, la de mayor valor comercial en val). "
             "Test solo aparece tras la evaluación única del congelamiento.", ""]
    if (rows["subset"].astype(float) < 1).any():
        lines += ["> **Atención:** incluye runs con un subconjunto de datos (pruebas de humo).", ""]
    return "\n".join(lines + _md_table(headers, body)) + "\n"


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

CSV_FRONT = ("iteration", "group", "name", "model", "notes", "n_runs", "seeds", "calibrated",
             "value_mismatch")
CSV_METRICS = (
    "subset", "params_total", "params_trainable", "size_mb", "train_acc", "val_f1_macro",
    "val_rate_busy_as_free",
    "val_value_comercial", "val_value_comercial_ci_lo", "val_value_comercial_ci_hi",
    "delta_value_comercial", "delta_ref_comercial", "val_value_comercial_1thr", "t_1thr_comercial",
    "operadores_extra_comercial", "t_L_comercial", "t_H_comercial", "val_verify_rate_comercial",
    "val_value_comercial_recomputed",
    "val_hours_uvg", "val_value_uvg", "val_value_uvg_ci_lo", "val_value_uvg_ci_hi",
    "delta_value_uvg", "delta_ref_uvg", "val_value_uvg_1thr", "t_1thr_uvg", "operadores_extra_uvg",
    "t_L_uvg", "t_H_uvg", "val_verify_rate_uvg", "val_value_uvg_recomputed",
    *(f"{sp}_{m}" for sp in EXTRA_SPLITS for m in ("f1_macro", "rate_busy_as_free", "value_comercial")),
    "temperature", "latency_ms", "cost_per_1000_pred_q", "inference_cost_month_q",
    "inference_cost_source", "train_time_s", "train_time_s_total", "train_cost_q",
    "train_cost_q_total", "train_cost_usd", "train_cost_usd_total",
)


def csv_columns(table: pd.DataFrame) -> list[str]:
    """Orden del CSV: identificación, métricas (cada una seguida de ``_std``) y el resto."""
    cols = [c for c in CSV_FRONT if c in table.columns]
    for col in CSV_METRICS:
        cols += [c for c in (col, f"{col}_std") if c in table.columns and c not in cols]
    cols += [c for c in table.columns if c not in cols and c != "run_ids"]
    return cols + (["run_ids"] if "run_ids" in table.columns else [])


def compute_rankings(records: Sequence[RunRecord], scenarios: Mapping[str, Mapping]) -> dict[str, pd.DataFrame]:
    reps = representatives(records)
    return {name: ranking_sensitivity(reps, name, cfg) for name, cfg in scenarios.items()}


def write_outputs(table: pd.DataFrame, out_dir: Path, runs_dir: Path, include_smoke: bool,
                  rankings: Mapping[str, pd.DataFrame] | None = None) -> list[Path]:
    """Escribe la tabla, el resumen y los CSV de ranking; devuelve las rutas escritas."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rankings = rankings or {}
    written = [out_dir / "iteraciones.csv", out_dir / "iteraciones.md", out_dir / "iteraciones_resumen.md"]
    table[csv_columns(table)].to_csv(written[0], index=False, encoding="utf-8-sig")
    written[1].write_text(render_markdown(table, runs_dir, include_smoke, rankings), encoding="utf-8")
    written[2].write_text(render_summary(table), encoding="utf-8")
    for name, rank in rankings.items():
        if not rank.empty:
            path = out_dir / f"ranking_sensibilidad_{name}.csv"
            rank.to_csv(path, index=False, encoding="utf-8-sig")
            written.append(path)
    return written


def generate(runs_dir: Path, out_dir: Path, include_smoke: bool = False,
             records: Sequence[RunRecord] | None = None) -> list[Path]:
    """Carga los runs (si no se pasan ya cargados) y escribe todas las salidas de la tabla."""
    if records is None:
        records = load_records(runs_dir, include_smoke)
    if not records:
        print(f"No hay runs completos en {runs_dir}; la tabla solo tendrá B0.")
    scenarios = load_scenarios()
    comparable_records(records, warn=True)
    table = build_table(records, scenarios)
    written = write_outputs(table, out_dir, runs_dir, include_smoke, compute_rankings(records, scenarios))
    print(f"{len(records)} run(s), {table.shape[0] - 1} fila(s) + B0 -> {out_dir}")
    for path in written:
        print(f"  {path.name}")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Junta los runs en la tabla de iteraciones.")
    parser.add_argument("--runs-dir", default=None, help="carpeta de runs (por defecto runs/)")
    parser.add_argument("--include-smoke", action="store_true",
                        help="incluye también runs/_smoke/ (pruebas de humo)")
    parser.add_argument("--out", default=None, help="carpeta de salida (por defecto results/)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs_dir = resolve_dir(args.runs_dir, paths.runs_dir())
    out_dir = resolve_dir(args.out, paths.results_dir())
    generate(runs_dir, out_dir, args.include_smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
