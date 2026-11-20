"""Costo de inferencia comparado entre versiones del modelo (B1, B2 y el modelo final).

Uso:
    python scripts/inference_costs.py [--models models/b1_hog_lr.joblib models/b2_malexnet.onnx models/best.onnx]
                                      [--n 300] [--n-frame N] [--no-gpu] [--skip-measure] [--ref b1_hog_lr]

1. Mide cada modelo con ``cupo.benchmark`` (mismo protocolo para todos, uno tras otro en la
   misma máquina para que la comparación sea justa) y guarda cada resultado en
   ``results/costo_inferencia/<name>.json``. Con ``--skip-measure`` reusa esos JSON. No escribe
   en ``models/`` ni en ``runs/``. La GPU se mide solo para modelos ONNX cuyo run (``run_id`` de
   la tarjeta ``<stem>.json``) existe en ``runs/`` y si hay CUDA.
2. Arma ``results/costo_inferencia.csv`` y ``results/costo_inferencia.md``: una fila por modelo
   con latencia, throughput y costo de inferencia (por 1,000 predicciones y al mes con el volumen
   del escenario comercial, ``value_model.inference_cost_period``), unida por ``name`` con
   ``results/iteraciones.csv`` (solo lectura) para el valor comercial en val, test y OOD y el
   costo de entrenamiento, más las diferencias frente al modelo de referencia (B1).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cupo import benchmark as B  # noqa: E402
from cupo import paths  # noqa: E402
from cupo.runs import load_json, save_json  # noqa: E402
from cupo.value_model import inference_cost_period, load_scenario  # noqa: E402

SCENARIO = "comercial"
DEFAULT_MODELS: tuple[str, ...] = ("b1_hog_lr.joblib", "b2_malexnet.onnx", "best.onnx")
DEFAULT_REF = "b1_hog_lr"
OUT_NAME = "costo_inferencia"
SPLITS: tuple[tuple[str, str], ...] = (("val", "validación"), ("test", "test"),
                                        ("ood", "otro parqueo (CNRPark A/B, OOD)"))

COLUMNS: tuple[str, ...] = (
    "name", "iteration", "label", "kind", "model", "file", "run_id", "params_total", "size_mb", "cpu",
    "cpu_1t_slot_p50_ms", "cpu_1t_frame_p50_ms", "cpu_1t_frame_throughput_pred_s",
    "cpu_all_slot_p50_ms", "cpu_all_frame_p50_ms", "preprocess_ms_per_patch", "pipeline_pred_s",
    "cost_per_1000_pred_q", "pred_month", "inference_cost_month_q",
    "gpu_device", "gpu_slot_p50_ms", "gpu_frame_p50_ms",
    "val_value_comercial", "test_value_comercial", "ood_value_comercial", "train_cost_q",
    "delta_val_value_q", "delta_test_value_q", "delta_ood_value_q", "delta_inference_cost_month_q",
    "delta_net_val_q", "delta_ref",
)


# ---------------------------------------------------------------------------
# Medición
# ---------------------------------------------------------------------------

def card_of(model_path: Path) -> dict[str, Any]:
    """Tarjeta ``<stem>.json`` del modelo (vacía si no existe)."""
    card_path = model_path.with_suffix(".json")
    return load_json(card_path) if card_path.exists() else {}


def model_name(model_path: Path) -> str:
    """``name`` de la tarjeta (el de ``iteraciones.csv``) o, si no hay, el stem del archivo."""
    return str(card_of(model_path).get("name") or model_path.stem)


def gpu_run_for(model_path: Path) -> str | None:
    """Run de PyTorch del modelo (para medir en GPU) si es ONNX y su carpeta existe en ``runs/``."""
    if B.model_kind(model_path) != B.KIND_ONNX:
        return None
    run_id = card_of(model_path).get("run_id")
    if not run_id or not (paths.runs_dir() / run_id).is_dir():
        return None
    try:
        B.check_run_matches(model_path, run_id)
    except (ValueError, FileNotFoundError):
        return None
    return run_id


def measure_model(model_path: Path, n: int, n_frame: int | None, no_gpu: bool) -> dict[str, Any]:
    """Benchmark de un modelo con el protocolo de ``cupo.benchmark`` (sin escribir en models/ ni runs/)."""
    run = None if no_gpu else gpu_run_for(model_path)
    args = argparse.Namespace(onnx=str(model_path), joblib=None, run=run, n=n, n_frame=n_frame,
                              no_gpu=no_gpu or run is None, out=None)
    return B.run_benchmark(args)


def bench_dir(results: Path) -> Path:
    return results / OUT_NAME


def measure_all(model_paths: Sequence[Path], results: Path, n: int, n_frame: int | None,
                no_gpu: bool) -> None:
    """Mide los modelos uno tras otro y guarda ``results/costo_inferencia/<name>.json``."""
    out_dir = bench_dir(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in model_paths:
        name = model_name(path)
        print(f"Midiendo {name} ({path.name})...", flush=True)
        res = measure_model(path, n, n_frame, no_gpu)
        save_json(out_dir / f"{name}.json", res)
        B.print_report(res)


# ---------------------------------------------------------------------------
# Tabla
# ---------------------------------------------------------------------------

def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _p50(block: Mapping[str, Any] | None, mode: str) -> float:
    return _num(((block or {}).get(mode) or {}).get("p50_ms"))


def bench_row(bench: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Columnas de inferencia de un ``benchmark.json`` + costo al mes del escenario."""
    info = bench.get("model") or {}
    cpu1, cpu_all, gpu = bench.get("cpu_1_thread"), bench.get("cpu_all_threads"), bench.get("gpu_torch")
    cost_1000 = _num(bench.get("costo_por_1000_pred_Q"))
    period = inference_cost_period(scenario_cfg, cost_per_1000=cost_1000)
    sfx = period["periodo"]
    return {
        "name": info.get("name") or Path(str(info.get("file") or info.get("onnx"))).stem,
        "kind": info.get("kind") or B.KIND_ONNX,
        "model": info.get("model"),
        "file": info.get("file") or info.get("onnx"),
        "run_id": info.get("run_id"),
        "size_mb": _num(info.get("size_mb")),
        "cpu": (bench.get("env") or {}).get("cpu"),
        "cpu_1t_slot_p50_ms": _p50(cpu1, "per_slot"),
        "cpu_1t_frame_p50_ms": _p50(cpu1, "per_frame"),
        "cpu_1t_frame_throughput_pred_s": _num(((cpu1 or {}).get("per_frame") or {}).get("throughput_pred_s")),
        "cpu_all_slot_p50_ms": _p50(cpu_all, "per_slot"),
        "cpu_all_frame_p50_ms": _p50(cpu_all, "per_frame"),
        "preprocess_ms_per_patch": _num(bench.get("preprocess_ms_per_patch")),
        "pipeline_pred_s": _num((bench.get("cost") or {}).get("pipeline_pred_s")),
        "cost_per_1000_pred_q": cost_1000,
        "pred_month": _num(period[f"pred_{sfx}"]),
        "inference_cost_month_q": _num(period[f"costo_inf_{sfx}"]),
        "gpu_device": (gpu or {}).get("device"),
        "gpu_slot_p50_ms": _p50(gpu, "per_slot"),
        "gpu_frame_p50_ms": _p50(gpu, "per_frame"),
    }


def iteration_row(it: pd.DataFrame, name: str) -> dict[str, Any]:
    """Valor comercial (val, test, OOD), parámetros y costo de entrenamiento de ``iteraciones.csv``."""
    match = it[it["name"] == name]
    if match.empty:
        return {}
    row = match.iloc[0]
    train_cost = row.get("train_cost_q_total")
    if pd.isna(train_cost):
        train_cost = row.get("train_cost_q")
    return {
        "iteration": row.get("iteration"),
        "params_total": _num(row.get("params_total")),
        "val_value_comercial": _num(row.get("val_value_comercial")),
        "test_value_comercial": _num(row.get("test_value_comercial")),
        "ood_value_comercial": _num(row.get("ood_value_comercial")),
        "train_cost_q": _num(train_cost),
    }


def model_label(row: Mapping[str, Any]) -> str:
    """Etiqueta legible: el archivo ``best.*`` es el modelo final."""
    stem = Path(str(row.get("file") or "")).stem
    if stem == "best":
        return f"Modelo final ({row['name']})"
    it = row.get("iteration")
    return f"{it} ({row['name']})" if isinstance(it, str) and it and it != row["name"] else str(row["name"])


def add_deltas(df: pd.DataFrame, ref: str) -> pd.DataFrame:
    """Diferencias frente al modelo ``ref`` (Q/mes). ``delta_net_val_q`` = Δ valor val − Δ costo de inferencia."""
    df = df.copy()
    base = df[df["name"] == ref]
    pairs = (("delta_val_value_q", "val_value_comercial"), ("delta_test_value_q", "test_value_comercial"),
             ("delta_ood_value_q", "ood_value_comercial"),
             ("delta_inference_cost_month_q", "inference_cost_month_q"))
    for delta, col in pairs:
        df[delta] = df[col] - base[col].iloc[0] if not base.empty else float("nan")
    df["delta_net_val_q"] = df["delta_val_value_q"] - df["delta_inference_cost_month_q"]
    df["delta_ref"] = ref if not base.empty else ""
    return df


def build_table(benches: Sequence[Mapping[str, Any]], iterations_csv: Path, ref: str) -> pd.DataFrame:
    """Una fila por modelo con inferencia, valor (de ``iteraciones.csv``) y diferencias frente a ``ref``."""
    scenario_cfg = load_scenario(SCENARIO)
    it = pd.read_csv(iterations_csv, encoding="utf-8-sig") if iterations_csv.is_file() else pd.DataFrame(
        columns=["name"])
    rows = []
    for bench in benches:
        row = bench_row(bench, scenario_cfg)
        row.update(iteration_row(it, row["name"]))
        row["label"] = model_label(row)
        rows.append(row)
    # Las columnas de valor existen aunque ningún modelo esté en iteraciones.csv (quedan en NaN)
    df = pd.DataFrame(rows).reindex(columns=list(dict.fromkeys(
        [*pd.DataFrame(rows).columns, "val_value_comercial", "test_value_comercial", "ood_value_comercial"])))
    df = add_deltas(df, ref)
    return df.reindex(columns=list(COLUMNS))


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _f(value: Any, digits: int = 2, signed: bool = False) -> str:
    v = _num(value)
    if math.isnan(v):
        return "—"
    return f"{v:+,.{digits}f}" if signed else f"{v:,.{digits}f}"


def _money(value: float, digits: int = 0) -> str:
    return f"Q{abs(value):,.{digits}f}"


def conclusion(row: Mapping[str, Any], ref_label: str) -> str:
    """Frase con los números de la fila: valor extra frente a la referencia y costo extra de cómputo."""
    parts = []
    for split, text in SPLITS:
        d = _num(row.get(f"delta_{split}_value_q"))
        if not math.isnan(d):
            parts.append(f"{_money(d)} {'más' if d >= 0 else 'menos'} en {text}")
    d_cost = _num(row.get("delta_inference_cost_month_q"))
    if not parts or math.isnan(d_cost):
        return ""
    if len(parts) > 1:
        value_text = ", ".join(parts[:-1]) + " y " + parts[-1]
    else:
        value_text = parts[0]
    cost_word = "un costo extra" if d_cost >= 0 else "un ahorro"
    net = _num(row.get("delta_net_val_q"))
    net_text = "" if math.isnan(net) else (
        f" En validación, descontado ese cómputo, la diferencia neta es de {_money(net)} al mes "
        f"{'a favor' if net >= 0 else 'en contra'} del modelo.")
    return (f"{row['label']} genera al mes, frente a {ref_label}, {value_text}, con {cost_word} de cómputo de "
            f"{_money(d_cost, 2)} al mes (inferencia en CPU con 1 núcleo para "
            f"{_num(row['pred_month']):,.0f} predicciones al mes del escenario comercial).{net_text}")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def inference_table(df: pd.DataFrame) -> list[str]:
    header = ["Modelo", "Parámetros", "Archivo (MB)", "CPU 1 hilo, espacio p50 (ms)",
              "CPU 1 hilo, frame de 46 p50 (ms)", "CPU todos los hilos, frame p50 (ms)",
              "Throughput 1 núcleo con preproc. (pred/s)", "Costo por 1,000 pred. (Q)",
              "Costo de inferencia (Q/mes)", "GPU, frame p50 (ms)"]
    rows = [[r["label"], _f(r["params_total"], 0), _f(r["size_mb"], 3), _f(r["cpu_1t_slot_p50_ms"]),
             _f(r["cpu_1t_frame_p50_ms"], 1), _f(r["cpu_all_frame_p50_ms"], 1), _f(r["pipeline_pred_s"], 0),
             _f(r["cost_per_1000_pred_q"], 6), _f(r["inference_cost_month_q"]), _f(r["gpu_frame_p50_ms"])]
            for _, r in df.iterrows()]
    return _table(header, rows)


def value_table(df: pd.DataFrame, ref_label: str) -> list[str]:
    header = ["Modelo", "Valor val (Q/mes)", "Valor test (Q/mes)", "Valor OOD (Q/mes)",
              "Costo de entrenamiento (Q)", f"Δ valor val vs {ref_label}", "Δ valor test", "Δ valor OOD",
              "Δ costo de inferencia (Q/mes)", "Δ neto val (Q/mes)"]
    rows = [[r["label"], _f(r["val_value_comercial"]), _f(r["test_value_comercial"]),
             _f(r["ood_value_comercial"]), _f(r["train_cost_q"], 3), _f(r["delta_val_value_q"], signed=True),
             _f(r["delta_test_value_q"], signed=True), _f(r["delta_ood_value_q"], signed=True),
             _f(r["delta_inference_cost_month_q"], signed=True), _f(r["delta_net_val_q"], signed=True)]
            for _, r in df.iterrows()]
    return _table(header, rows)


def render_markdown(df: pd.DataFrame, benches: Sequence[Mapping[str, Any]], ref: str) -> str:
    ref_rows = df[df["name"] == ref]
    ref_label = str(ref_rows["label"].iloc[0]) if not ref_rows.empty else ref
    cpus = sorted({str(c) for c in df["cpu"].dropna()})
    first = benches[0] if benches else {}
    costs = (first.get("cost") or {})
    pred_month = _num(df["pred_month"].iloc[0]) if len(df) else float("nan")
    lines = [
        "# Costo de inferencia por modelo",
        "",
        "Generado por `python scripts/inference_costs.py`. Cada modelo se midió con `cupo.benchmark` "
        f"(mismo protocolo, uno tras otro en la misma máquina: {', '.join(cpus) or '—'}) con "
        f"{first.get('data', {}).get('n_images', '—')} patches de val (semilla {first.get('data', {}).get('seed', '—')}), "
        f"{first.get('n_slot', '—')} repeticiones por espacio y {first.get('n_frame', '—')} por frame. "
        "Los JSON de cada medición están en `results/costo_inferencia/`.",
        "",
        "Los modelos se midieron en la misma sesión para que la comparación entre ellos sea justa. El valor "
        "absoluto de la latencia y del costo cambia entre sesiones y entre máquinas (por ejemplo, "
        "`models/benchmark.json` del modelo final se midió en otra máquina); la diferencia entre modelos y la "
        "conclusión frente a la referencia son lo que se compara.",
        "",
        "## Latencia y costo de inferencia",
        "",
        *inference_table(df),
        "",
        "- La latencia por espacio es batch 1; la del frame, un batch de 46 espacios. CPU con ONNX Runtime "
        "(B2 y modelo final) o con scikit-image + scikit-learn (B1).",
        "- En B1 la latencia del modelo incluye el cálculo de HOG (su extracción de características: gris, "
        "64x64 y `skimage.feature.hog`), con el mismo camino que la app; su preprocesamiento es solo abrir y "
        "decodificar la imagen. En las CNN el preprocesamiento (`preprocess_numpy`) se mide aparte.",
        "- Costo por 1,000 predicciones: tiempo de 1 núcleo en modo frame, modelo + preprocesamiento, a "
        f"USD {_f(costs.get('cpu_server_usd_per_hour'), 2)}/h por núcleo y Q{_f(costs.get('gtq_per_usd'), 2)} por "
        "USD (`configs/costs.yaml`).",
        f"- Costo de inferencia al mes: {pred_month:,.0f} predicciones al mes del escenario comercial "
        "(`configs/value/comercial.yaml`: espacios · 3600/refresh_s · horas de operación · días) × costo por "
        "1,000 / 1,000, con `value_model.inference_cost_period`.",
        "- GPU: PyTorch con CUDA (AMP si el run lo usaba); solo para los modelos de PyTorch con su run local.",
        "",
        "## Valor de negocio frente al costo",
        "",
        *value_table(df, ref_label),
        "",
        "- Valor comercial (Q/mes para el operador), parámetros y costo de entrenamiento: de "
        "`results/iteraciones.csv`. Test y OOD se evaluaron una sola vez, después del congelamiento.",
        "- Ese valor descuenta para todos los modelos el mismo costo de inferencia por defecto "
        "(`costo_por_1000_pred_Q` de `configs/costs.yaml`), así que Δ valor no incluye la diferencia de cómputo; "
        "Δ neto val = Δ valor val − Δ costo de inferencia medido.",
        "",
        "## Conclusión",
        "",
    ]
    for _, r in df.iterrows():
        if r["name"] == ref:
            continue
        text = conclusion(r, ref_label)
        if text:
            lines += [f"- {text}", ""]
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Costo de inferencia comparado entre modelos (misma máquina).")
    parser.add_argument("--models", nargs="+", default=[str(paths.models_dir() / m) for m in DEFAULT_MODELS],
                        help="modelos a medir (.onnx o .joblib, cada uno con su tarjeta <stem>.json)")
    parser.add_argument("--out", default=str(paths.results_dir()),
                        help="carpeta de costo_inferencia.csv/.md y de costo_inferencia/<name>.json")
    parser.add_argument("--iterations", default=None,
                        help="tabla de iteraciones (por defecto results/iteraciones.csv; solo lectura)")
    parser.add_argument("--ref", default=DEFAULT_REF, help="name del modelo de referencia para las diferencias")
    parser.add_argument("--n", type=int, default=300, help="repeticiones por espacio (batch 1)")
    parser.add_argument("--n-frame", type=int, default=None, help="repeticiones por frame (por defecto min(n, 100))")
    parser.add_argument("--no-gpu", action="store_true", help="omite la medición en GPU")
    parser.add_argument("--skip-measure", action="store_true",
                        help="no mide: reusa los JSON de costo_inferencia/ y solo arma la tabla")
    return parser.parse_args(argv)


def load_benches(model_paths: Sequence[Path], results: Path) -> list[dict[str, Any]]:
    """JSON de cada modelo en ``costo_inferencia/``; ``FileNotFoundError`` si falta alguno."""
    out = []
    for path in model_paths:
        target = bench_dir(results) / f"{model_name(path)}.json"
        if not target.is_file():
            raise FileNotFoundError(f"No existe costo_inferencia/{target.name}; corre sin --skip-measure.")
        out.append(load_json(target))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_paths = [Path(m).resolve() for m in args.models]
    missing = [p.name for p in model_paths if not p.is_file()]
    if missing:
        print(f"Error: no existen los modelos {missing} (ver la sección de exportación del README)", file=sys.stderr)
        return 1
    results = Path(args.out)
    iterations = Path(args.iterations) if args.iterations else paths.results_dir() / "iteraciones.csv"
    if not args.skip_measure:
        measure_all(model_paths, results, args.n, args.n_frame, args.no_gpu)
    try:
        benches = load_benches(model_paths, results)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    df = build_table(benches, iterations, args.ref)
    results.mkdir(parents=True, exist_ok=True)
    df.to_csv(results / f"{OUT_NAME}.csv", index=False, encoding="utf-8")
    md = render_markdown(df, benches, args.ref)
    (results / f"{OUT_NAME}.md").write_text(md, encoding="utf-8")
    # La consola de Windows puede no tener Δ o −: se reemplazan al imprimir, no en los archivos
    sys.stdout.reconfigure(errors="replace")
    print(md)
    print(f"Escrito en: {results / (OUT_NAME + '.csv')}, {results / (OUT_NAME + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
