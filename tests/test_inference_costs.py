"""Costo de inferencia por modelo.

- ``value_model.inference_cost_period``: casos a mano (escenario sintético mensual y semestral,
  comercial con el costo por defecto = Q165.312/mes, escala lineal con ``cost_per_1000``) y
  coherencia con ``compute_value``; ``value_val.json`` de los runs de B1 se reproduce igual.
- ``scripts/inference_costs.py`` con ``--skip-measure`` y JSON/``iteraciones.csv`` sintéticos:
  columnas, diferencias frente a la referencia calculadas a mano, frase de conclusión, modelos
  que no están en ``iteraciones.csv`` y que no escribe en ``models/`` ni en ``results/iteraciones.*``.
- Medición corta sin dataset (patches sintéticos) y, si existen, que las tablas versionadas en
  ``results/`` se reconstruyen igual desde sus JSON.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import make_rates
from cupo import benchmark as B
from cupo import classical as CL
from cupo import paths
from cupo import value_model as vm
from cupo.metrics import three_state_rates

from run_helpers import REPO

SCRIPT = REPO / "scripts" / "inference_costs.py"
RESULTS = REPO / "results"
MODELS_DIR = REPO / "models"
PRED_MONTH_COM = 164 * (3600 // 30) * 14 * 30  # 8,265,600 predicciones al mes


def load_script():
    name = "cupo_scripts_inference_costs"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


IC = load_script()


def _stamp(files) -> list[tuple[str, int]]:
    return sorted((p.name, p.stat().st_mtime_ns) for p in files if p.exists())


def _protected_stamp() -> dict[str, list]:
    """models/, results/iteraciones.* y las tablas versionadas de costo de inferencia."""
    return {
        "models": _stamp(MODELS_DIR.iterdir()) if MODELS_DIR.is_dir() else [],
        "iteraciones": _stamp(RESULTS.glob("iteraciones*")),
        "costo": _stamp([*RESULTS.glob("costo_inferencia*"), *(RESULTS / "costo_inferencia").glob("*")]),
    }


# ===========================================================================
# value_model.inference_cost_period
# ===========================================================================

def _scenario(periodo: str, params: dict) -> dict:
    return {"name": "sintetico", "periodo": periodo, "objective": f"valor_operador_{periodo}", "params": params}


def test_inference_cost_period_synthetic_month_by_hand():
    cfg = _scenario("mes", {"espacios": {"value": 10, "source": "x", "confidence": "estimado"},
                            "refresh_s": 60, "horas_operacion_dia": 8, "dias_mes": 20,
                            "costo_por_1000_pred_Q": 0.5})
    out = vm.inference_cost_period(cfg)
    # 10 espacios · 60 frames/h · 8 h · 20 días = 96,000; 96,000 / 1000 · 0.5 = Q48
    assert set(out) == {"periodo", "costo_por_1000_pred_Q", "pred_mes", "costo_inf_mes"}
    assert out["periodo"] == "mes"
    assert out["pred_mes"] == pytest.approx(96_000.0)
    assert out["costo_por_1000_pred_Q"] == 0.5
    assert out["costo_inf_mes"] == pytest.approx(48.0)
    # Con costo explícito sustituye al del escenario
    out2 = vm.inference_cost_period(cfg, cost_per_1000=0.125)
    assert out2["costo_por_1000_pred_Q"] == 0.125 and out2["costo_inf_mes"] == pytest.approx(12.0)
    assert cfg["params"]["costo_por_1000_pred_Q"] == 0.5


def test_inference_cost_period_semester_uses_default_cost():
    cfg = _scenario("semestre", {"espacios": 4, "refresh_s": 30, "horas_operacion_dia": 10,
                                 "dias_semestre": 50})
    out = vm.inference_cost_period(cfg)
    # 4 · 120 · 10 · 50 = 240,000; sin costo en el escenario → configs/costs.yaml (Q0.02)
    assert out["periodo"] == "semestre"
    assert out["pred_semestre"] == pytest.approx(240_000.0)
    assert out["costo_por_1000_pred_Q"] == pytest.approx(vm.default_cost_per_1000())
    assert out["costo_inf_semestre"] == pytest.approx(240.0 * vm.default_cost_per_1000())


def test_inference_cost_commercial_default_by_hand(com):
    assert vm.default_cost_per_1000() == 0.02
    out = vm.inference_cost_period(com)
    assert out["periodo"] == "mes"
    assert out["pred_mes"] == PRED_MONTH_COM == 8_265_600
    # 8,265,600 / 1000 · 0.02 = 165.312 exacto
    assert Fraction(out["costo_inf_mes"]).limit_denominator(10**6) == Fraction(165312, 1000)
    assert out["costo_inf_mes"] == pytest.approx(165.312, abs=1e-9)


@pytest.mark.parametrize("c", [0.0, 0.0002187425759835728, 0.001, 0.05, 1.0])
def test_inference_cost_scales_linearly(com, c):
    out = vm.inference_cost_period(com, cost_per_1000=c)
    assert out["costo_por_1000_pred_Q"] == c
    assert out["pred_mes"] == PRED_MONTH_COM
    assert out["costo_inf_mes"] == pytest.approx(PRED_MONTH_COM / 1000 * c, rel=1e-12, abs=1e-12)
    base = vm.inference_cost_period(com, cost_per_1000=0.02)["costo_inf_mes"]
    assert out["costo_inf_mes"] == pytest.approx(base * c / 0.02, rel=1e-12, abs=1e-12)


def test_inference_cost_005_by_hand(com):
    assert vm.inference_cost_period(com, cost_per_1000=0.05)["costo_inf_mes"] == pytest.approx(413.28)


def test_inference_cost_period_does_not_mutate(com):
    before = copy.deepcopy(com)
    vm.inference_cost_period(com, cost_per_1000=7.0)
    assert com == before


@pytest.mark.parametrize("scenario", ["com", "uvg"])
@pytest.mark.parametrize("c", [None, 0.0004])
def test_inference_cost_matches_compute_value(scenario, c, request):
    cfg = request.getfixturevalue(scenario)
    if c is not None:
        cfg = vm.with_param(cfg, vm.COST_PER_1000_KEY, c)
    rates = make_rates(0.01, 0.03, 0.85, 0.05)
    full = vm.compute_value(rates, cfg)
    part = vm.inference_cost_period(cfg)
    sfx = part["periodo"]
    assert part[f"pred_{sfx}"] == pytest.approx(full[f"pred_{sfx}"], rel=1e-12)
    assert part[f"costo_inf_{sfx}"] == pytest.approx(full[f"costo_inf_{sfx}"], rel=1e-12)
    assert part["costo_por_1000_pred_Q"] == pytest.approx(full["costo_por_1000_pred_Q"], rel=1e-12)


def _b1_runs() -> list[Path]:
    runs = sorted((REPO / "runs").glob("b1_*")) if (REPO / "runs").is_dir() else []
    return [r for r in runs if (r / "value_val.json").is_file() and (r / "val_preds.npz").is_file()]


@pytest.mark.skipif(not _b1_runs(), reason="no hay runs/b1_* con value_val.json y val_preds.npz")
@pytest.mark.parametrize("run_dir", _b1_runs(), ids=lambda p: p.name)
def test_value_val_of_b1_recomputes_identically(run_dir):
    """El valor en val de B1 (umbrales guardados) se reproduce igual con el modelo de valor actual."""
    saved = json.loads((run_dir / "value_val.json").read_text(encoding="utf-8"))
    preds = np.load(run_dir / "val_preds.npz", allow_pickle=False)
    for scenario, entry in saved.items():
        rates = three_state_rates(preds["y"], preds["p"], entry["t_L"], entry["t_H"])
        comp = vm.compute_value(rates, vm.load_scenario(scenario))
        assert comp["objective"] == pytest.approx(entry["objective_value"], rel=1e-9), scenario
        checked = 0
        for key, val in entry["components"].items():
            if isinstance(val, bool) or not isinstance(val, (int, float)) or key not in comp:
                continue
            if isinstance(val, float) and math.isnan(val):
                continue
            assert comp[key] == pytest.approx(val, rel=1e-9, abs=1e-12), (scenario, key)
            checked += 1
        assert checked > 20
        sfx = vm.load_scenario(scenario)["periodo"]
        inf = vm.inference_cost_period(vm.load_scenario(scenario))
        assert inf[f"costo_inf_{sfx}"] == pytest.approx(entry["components"][f"costo_inf_{sfx}"], rel=1e-12)
        assert inf[f"pred_{sfx}"] == pytest.approx(entry["components"][f"pred_{sfx}"], rel=1e-12)


# ===========================================================================
# scripts/inference_costs.py (sin medir)
# ===========================================================================

# Costo por 1,000 predicciones (Q) de cada benchmark sintético
COST_1000 = {"b1_hog_lr": 0.0002, "b2_malexnet": 0.0005, "final_x": 0.001, "nuevo": 0.0004}
# Valor comercial (Q/mes) de iteraciones.csv sintético: val, test, OOD
VALUES = {"b1_hog_lr": (50_000.0, 49_000.0, 45_000.0),
          "b2_malexnet": (50_400.0, 49_100.0, 46_000.0),
          "final_x": (50_900.0, 48_800.0, 47_000.0)}


def _bench(name: str | None, file: str, kind: str, cost: float, gpu: bool = False) -> dict:
    block = {"per_slot": {"p50_ms": 2.5, "throughput_pred_s": 400.0},
             "per_frame": {"p50_ms": 80.0, "throughput_pred_s": 575.0}}
    return {
        "model": {"name": name, "file": file, "kind": kind, "onnx": file if kind == "onnx" else None,
                  "model": "m", "run_id": f"{name}_run" if name else None, "size_mb": 0.25},
        "data": {"split": "val", "n_images": 46, "seed": 0}, "n_slot": 300, "n_frame": 100,
        "preprocess_ms_per_patch": 1.5,
        "cpu_1_thread": block, "cpu_all_threads": block,
        "gpu_torch": ({"device": "GPU de prueba", "per_slot": {"p50_ms": 1.0}, "per_frame": {"p50_ms": 4.0}}
                      if gpu else None),
        "costo_por_1000_pred_Q": cost,
        "cost": {"q": cost, "pipeline_pred_s": 500.0, "cpu_server_usd_per_hour": 0.05, "gtq_per_usd": 7.7},
        "env": {"cpu": "CPU de prueba"},
    }


@pytest.fixture
def setup(tmp_path):
    """Modelos (archivos vacíos + tarjetas), JSON medidos e iteraciones.csv sintéticos."""
    models = tmp_path / "models"
    models.mkdir()
    specs = [("b1_hog_lr.joblib", "b1_hog_lr", "classical"), ("b2_malexnet.onnx", "b2_malexnet", "onnx"),
             ("best.onnx", "final_x", "onnx"), ("nuevo.onnx", None, "onnx")]
    model_paths = []
    for file, name, _ in specs:
        p = models / file
        p.write_bytes(b"modelo")
        if name:
            p.with_suffix(".json").write_text(json.dumps({"name": name, "run_id": f"{name}_run"}),
                                              encoding="utf-8")
        model_paths.append(p)
    out = tmp_path / "results"
    (out / "costo_inferencia").mkdir(parents=True)
    for file, name, kind in specs:
        key = name or Path(file).stem
        bench = _bench(name, file, kind, COST_1000[key], gpu=(name == "final_x"))
        (out / "costo_inferencia" / f"{key}.json").write_text(json.dumps(bench), encoding="utf-8")
    it = pd.DataFrame([
        {"iteration": "B0", "name": "b0_status_quo", "params_total": None, "val_value_comercial": 0.0,
         "test_value_comercial": None, "ood_value_comercial": None, "train_cost_q": None,
         "train_cost_q_total": None},
        {"iteration": "B1", "name": "b1_hog_lr", "params_total": 1765, "val_value_comercial": VALUES["b1_hog_lr"][0],
         "test_value_comercial": VALUES["b1_hog_lr"][1], "ood_value_comercial": VALUES["b1_hog_lr"][2],
         "train_cost_q": 0.01, "train_cost_q_total": 0.01},
        {"iteration": "B2", "name": "b2_malexnet", "params_total": 42460,
         "val_value_comercial": VALUES["b2_malexnet"][0], "test_value_comercial": VALUES["b2_malexnet"][1],
         "ood_value_comercial": VALUES["b2_malexnet"][2], "train_cost_q": 0.5, "train_cost_q_total": None},
        {"iteration": "final_x", "name": "final_x", "params_total": 1_500_000,
         "val_value_comercial": VALUES["final_x"][0], "test_value_comercial": VALUES["final_x"][1],
         "ood_value_comercial": VALUES["final_x"][2], "train_cost_q": 0.2, "train_cost_q_total": 0.6},
    ])
    it_csv = tmp_path / "iteraciones.csv"
    it.to_csv(it_csv, index=False, encoding="utf-8-sig")
    return {"root": tmp_path, "models": models, "paths": model_paths, "out": out, "it": it_csv}


def _main(setup, extra: list[str] | None = None, models: list[Path] | None = None) -> int:
    ms = [str(p) for p in (models or setup["paths"])]
    return IC.main(["--models", *ms, "--skip-measure", "--out", str(setup["out"]),
                    "--iterations", str(setup["it"]), *(extra or [])])


def test_parse_args_defaults():
    args = IC.parse_args([])
    assert [Path(m) for m in args.models] == [paths.models_dir() / m for m in
                                              ("b1_hog_lr.joblib", "b2_malexnet.onnx", "best.onnx")]
    assert Path(args.out) == paths.results_dir()
    assert args.ref == "b1_hog_lr" and args.iterations is None and args.skip_measure is False
    assert args.n == 300 and args.n_frame is None and args.no_gpu is False


def test_skip_measure_csv_by_hand(setup, capsys):
    before = _protected_stamp()
    models_before = _stamp(setup["models"].iterdir())
    benches_before = _stamp((setup["out"] / "costo_inferencia").iterdir())
    it_before = setup["it"].read_bytes()
    assert _main(setup) == 0
    capsys.readouterr()

    out = setup["out"]
    assert sorted(p.name for p in out.iterdir()) == ["costo_inferencia", "costo_inferencia.csv",
                                                     "costo_inferencia.md"]
    df = pd.read_csv(out / "costo_inferencia.csv")
    assert list(df.columns) == list(IC.COLUMNS)
    assert list(df["name"]) == ["b1_hog_lr", "b2_malexnet", "final_x", "nuevo"]
    r = df.set_index("name")

    # Costo al mes = 8,265,600 / 1000 · costo por 1000
    for name, c in COST_1000.items():
        assert r.loc[name, "pred_month"] == PRED_MONTH_COM
        assert r.loc[name, "cost_per_1000_pred_q"] == pytest.approx(c)
        assert r.loc[name, "inference_cost_month_q"] == pytest.approx(PRED_MONTH_COM / 1000 * c, rel=1e-12)
    assert r.loc["b1_hog_lr", "inference_cost_month_q"] == pytest.approx(1.65312)
    assert r.loc["b2_malexnet", "inference_cost_month_q"] == pytest.approx(4.1328)
    assert r.loc["final_x", "inference_cost_month_q"] == pytest.approx(8.2656)
    assert r.loc["nuevo", "inference_cost_month_q"] == pytest.approx(3.30624)

    # Diferencias frente a B1, a mano
    expected = {
        "b1_hog_lr": (0.0, 0.0, 0.0, 0.0, 0.0),
        "b2_malexnet": (400.0, 100.0, 1_000.0, 2.47968, 397.52032),
        "final_x": (900.0, -200.0, 2_000.0, 6.61248, 893.38752),
    }
    cols = ("delta_val_value_q", "delta_test_value_q", "delta_ood_value_q", "delta_inference_cost_month_q",
            "delta_net_val_q")
    for name, vals in expected.items():
        for col, v in zip(cols, vals):
            assert r.loc[name, col] == pytest.approx(v, abs=1e-9), (name, col)
    assert set(df["delta_ref"]) == {"b1_hog_lr"}

    # Datos de iteraciones.csv
    assert r.loc["b1_hog_lr", "iteration"] == "B1" and r.loc["b2_malexnet", "iteration"] == "B2"
    assert r.loc["b2_malexnet", "params_total"] == 42460
    assert r.loc["b2_malexnet", "train_cost_q"] == pytest.approx(0.5)   # sin total → train_cost_q
    assert r.loc["final_x", "train_cost_q"] == pytest.approx(0.6)       # train_cost_q_total
    assert r.loc["final_x", "val_value_comercial"] == VALUES["final_x"][0]
    assert r.loc["final_x", "test_value_comercial"] == VALUES["final_x"][1]
    assert r.loc["final_x", "ood_value_comercial"] == VALUES["final_x"][2]

    # Etiquetas y columnas de inferencia
    assert r.loc["b1_hog_lr", "label"] == "B1 (b1_hog_lr)"
    assert r.loc["final_x", "label"] == "Modelo final (final_x)"
    assert r.loc["b1_hog_lr", "kind"] == "classical" and r.loc["b2_malexnet", "kind"] == "onnx"
    assert r.loc["b1_hog_lr", "file"] == "b1_hog_lr.joblib"
    assert r.loc["final_x", "gpu_device"] == "GPU de prueba" and r.loc["final_x", "gpu_frame_p50_ms"] == 4.0
    assert pd.isna(r.loc["b1_hog_lr", "gpu_device"]) and pd.isna(r.loc["b1_hog_lr", "gpu_frame_p50_ms"])
    assert r.loc["b2_malexnet", "cpu_1t_frame_p50_ms"] == 80.0
    assert r.loc["b2_malexnet", "cpu_1t_frame_throughput_pred_s"] == 575.0
    assert r.loc["b2_malexnet", "pipeline_pred_s"] == 500.0

    # No escribe en models/ (ni en el real ni en el de la prueba), ni en iteraciones.*, ni en
    # los JSON medidos; las tablas versionadas de results/ no cambian
    assert _protected_stamp() == before
    assert _stamp(setup["models"].iterdir()) == models_before
    assert _stamp((out / "costo_inferencia").iterdir()) == benches_before
    assert setup["it"].read_bytes() == it_before


def test_model_not_in_iterations(setup, capsys):
    assert _main(setup) == 0
    capsys.readouterr()
    df = pd.read_csv(setup["out"] / "costo_inferencia.csv")
    row = df[df["name"] == "nuevo"].iloc[0]
    assert row["label"] == "nuevo" and row["file"] == "nuevo.onnx"
    for col in ("iteration", "params_total", "val_value_comercial", "test_value_comercial",
                "ood_value_comercial", "train_cost_q", "delta_val_value_q", "delta_test_value_q",
                "delta_ood_value_q", "delta_net_val_q"):
        assert pd.isna(row[col]), col
    # El costo de inferencia sí se compara: 3.30624 − 1.65312
    assert row["delta_inference_cost_month_q"] == pytest.approx(1.65312)
    md = (setup["out"] / "costo_inferencia.md").read_text(encoding="utf-8")
    # Segunda tabla (valor de negocio frente al costo)
    line = [ln for ln in md.splitlines() if ln.startswith("| nuevo |")][-1]
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[1] == "—" and cells[5] == "—" and cells[8] == "+1.65" and cells[9] == "—"
    assert "- nuevo genera" not in md


def test_markdown_conclusion_by_hand(setup, capsys):
    assert _main(setup) == 0
    printed = capsys.readouterr().out
    md = (setup["out"] / "costo_inferencia.md").read_text(encoding="utf-8")
    tail = ("(inferencia en CPU con 1 núcleo para 8,265,600 predicciones al mes del escenario comercial).")
    b2 = ("B2 (b2_malexnet) genera al mes, frente a B1 (b1_hog_lr), Q400 más en validación, Q100 más en test "
          "y Q1,000 más en otro parqueo (CNRPark A/B, OOD), con un costo extra de cómputo de Q2.48 al mes "
          f"{tail} En validación, descontado ese cómputo, la diferencia neta es de Q398 al mes a favor del modelo.")
    final = ("Modelo final (final_x) genera al mes, frente a B1 (b1_hog_lr), Q900 más en validación, Q200 menos "
             "en test y Q2,000 más en otro parqueo (CNRPark A/B, OOD), con un costo extra de cómputo de Q6.61 "
             f"al mes {tail} En validación, descontado ese cómputo, la diferencia neta es de Q893 al mes a favor "
             "del modelo.")
    conclusion = md[md.index("## Conclusión"):]
    bullets = [ln[2:] for ln in conclusion.splitlines() if ln.startswith("- ")]
    assert bullets == [b2, final]
    # Tabla de valor con los Δ a mano
    line = next(ln for ln in md.splitlines() if ln.startswith("| Modelo final (final_x) |") and "+900.00" in ln)
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[1:] == ["50,900.00", "48,800.00", "47,000.00", "0.600", "+900.00", "-200.00", "+2,000.00",
                         "+6.61", "+893.39"]
    # Tabla de inferencia: costo por 1,000 con 6 decimales y Q/mes
    line = next(ln for ln in md.splitlines() if ln.startswith("| B1 (b1_hog_lr) |") and "0.000200" in ln)
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[1] == "1,765" and cells[7] == "0.000200" and cells[8] == "1.65" and cells[9] == "—"
    assert "8,265,600 predicciones al mes" in md
    assert "USD 0.05/h" in md and "Q7.70 por USD" in md
    assert "Δ valor val vs B1 (b1_hog_lr)" in md
    # Lo impreso es el mismo informe
    assert "Conclusión" in printed and "Escrito en:" in printed


def test_conclusion_cheaper_and_worse_model():
    row = {"label": "Otro (x)", "pred_month": 1_000_000.0, "delta_val_value_q": -60.0,
           "delta_test_value_q": float("nan"), "delta_ood_value_q": float("nan"),
           "delta_inference_cost_month_q": -0.25, "delta_net_val_q": -59.75}
    text = IC.conclusion(row, "B1 (b1_hog_lr)")
    assert text == ("Otro (x) genera al mes, frente a B1 (b1_hog_lr), Q60 menos en validación, con un ahorro de "
                    "cómputo de Q0.25 al mes (inferencia en CPU con 1 núcleo para 1,000,000 predicciones al mes "
                    "del escenario comercial). En validación, descontado ese cómputo, la diferencia neta es de "
                    "Q60 al mes en contra del modelo.")
    row["delta_inference_cost_month_q"] = float("nan")
    assert IC.conclusion(row, "B1") == ""


def test_other_reference(setup, capsys):
    assert _main(setup, ["--ref", "b2_malexnet"]) == 0
    capsys.readouterr()
    r = pd.read_csv(setup["out"] / "costo_inferencia.csv").set_index("name")
    assert r.loc["final_x", "delta_val_value_q"] == pytest.approx(500.0)
    assert r.loc["final_x", "delta_test_value_q"] == pytest.approx(-300.0)
    assert r.loc["final_x", "delta_inference_cost_month_q"] == pytest.approx(8.2656 - 4.1328)
    assert r.loc["final_x", "delta_net_val_q"] == pytest.approx(500.0 - 4.1328)
    assert r.loc["b1_hog_lr", "delta_val_value_q"] == pytest.approx(-400.0)
    assert set(r["delta_ref"]) == {"b2_malexnet"}


def test_unknown_reference_gives_nan_deltas(setup, capsys):
    assert _main(setup, ["--ref", "no_existe"]) == 0
    capsys.readouterr()
    df = pd.read_csv(setup["out"] / "costo_inferencia.csv")
    assert df["delta_val_value_q"].isna().all() and df["delta_net_val_q"].isna().all()
    md = (setup["out"] / "costo_inferencia.md").read_text(encoding="utf-8")
    conclusion = md[md.index("## Conclusión"):]
    assert not [ln for ln in conclusion.splitlines() if ln.startswith("- ")]


def test_only_models_not_in_iterations(setup, capsys):
    """Ningún modelo pedido está en iteraciones.csv: la tabla sale igual, sin valor ni Δ valor."""
    rc = _main(setup, ["--ref", "nuevo"], models=[setup["paths"][3]])
    capsys.readouterr()
    assert rc == 0
    df = pd.read_csv(setup["out"] / "costo_inferencia.csv")
    assert list(df.columns) == list(IC.COLUMNS) and list(df["name"]) == ["nuevo"]
    assert pd.isna(df.loc[0, "val_value_comercial"])
    assert df.loc[0, "inference_cost_month_q"] == pytest.approx(3.30624)


def test_missing_iterations_file(setup, capsys):
    """Sin iteraciones.csv (ruta inexistente) la tabla de inferencia se arma igual."""
    setup["it"].unlink()
    rc = _main(setup)
    capsys.readouterr()
    assert rc == 0
    df = pd.read_csv(setup["out"] / "costo_inferencia.csv")
    assert list(df.columns) == list(IC.COLUMNS)
    assert df["val_value_comercial"].isna().all()
    assert df["inference_cost_month_q"].notna().all()


def test_skip_measure_missing_json_fails(setup, capsys):
    (setup["out"] / "costo_inferencia" / "final_x.json").unlink()
    rc = _main(setup)
    err = capsys.readouterr().err
    assert rc == 1 and "final_x.json" in err
    assert not (setup["out"] / "costo_inferencia.csv").exists()
    assert not (setup["out"] / "costo_inferencia.md").exists()


def test_missing_model_file_fails(setup, capsys):
    rc = _main(setup, models=[setup["paths"][0], setup["models"] / "fantasma.onnx"])
    err = capsys.readouterr().err
    assert rc == 1 and "fantasma.onnx" in err
    assert not (setup["out"] / "costo_inferencia.csv").exists()


def test_script_never_reads_test_or_ood(setup, monkeypatch, capsys):
    """Armar la tabla no carga particiones: el valor de test/OOD viene de iteraciones.csv."""
    import cupo.data

    def boom(*a, **k):
        raise AssertionError("no debe cargar particiones")

    monkeypatch.setattr(cupo.data, "load_split", boom)
    assert _main(setup) == 0
    capsys.readouterr()


# ===========================================================================
# Medición (sin dataset: patches sintéticos)
# ===========================================================================

def test_gpu_run_for(tmp_path, monkeypatch):
    j = tmp_path / "b1.joblib"
    j.write_bytes(b"x")
    j.with_suffix(".json").write_text(json.dumps({"name": "b1", "run_id": "r1"}), encoding="utf-8")
    assert IC.gpu_run_for(j) is None                       # clásico: sin GPU
    o = tmp_path / "m.onnx"
    o.write_bytes(b"x")
    assert IC.gpu_run_for(o) is None                       # sin tarjeta
    o.with_suffix(".json").write_text(json.dumps({"name": "m", "run_id": "r_onnx"}), encoding="utf-8")
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(paths, "runs_dir", lambda: runs)
    assert IC.gpu_run_for(o) is None                       # el run no existe
    (runs / "r_onnx").mkdir()
    assert IC.gpu_run_for(o) == "r_onnx"
    assert IC.model_name(o) == "m" and IC.model_name(tmp_path / "sin_tarjeta.onnx") == "sin_tarjeta"


def test_measure_model_args(tmp_path, monkeypatch):
    seen = {}

    def fake(args):
        seen.update(vars(args))
        return {"ok": True}

    monkeypatch.setattr(B, "run_benchmark", fake)
    o = tmp_path / "m.onnx"
    o.write_bytes(b"x")
    assert IC.measure_model(o, n=5, n_frame=3, no_gpu=False) == {"ok": True}
    assert seen["onnx"] == str(o) and seen["joblib"] is None and seen["run"] is None
    assert seen["no_gpu"] is True and seen["n"] == 5 and seen["n_frame"] == 3


def test_measure_classical_end_to_end(tmp_path, monkeypatch, capsys):
    """Mide un .joblib pequeño con el protocolo de cupo.benchmark y arma la tabla."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    imgs = []
    for i in range(6):
        arr = rng.integers(0, 256, (150, 150, 3), dtype=np.uint8)
        if i % 2:
            arr[40:110, 30:120] = 25
        p = tmp_path / "patches" / f"p{i}.jpg"
        p.parent.mkdir(exist_ok=True)
        Image.fromarray(arr).save(p)
        imgs.append(str(p))
    hog = dict(CL.HOG_DEFAULTS)
    X = CL.hog_features(imgs, hog, n_jobs=1)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(X, [i % 2 for i in range(6)])
    models = tmp_path / "models"
    models.mkdir()
    model = models / "hog_tiny.joblib"
    joblib.dump({"pipeline": pipe, "hog": hog}, model)
    model.with_suffix(".json").write_text(json.dumps({"name": "hog_tiny", "run_id": "hog_tiny_s0"}),
                                          encoding="utf-8")
    it = tmp_path / "it.csv"
    pd.DataFrame([{"iteration": "B1", "name": "hog_tiny", "val_value_comercial": 1000.0}]).to_csv(it, index=False)
    monkeypatch.setattr(B, "sample_val_paths", lambda n_images=46, seed=0: list(imgs))

    before = _protected_stamp()
    models_before = _stamp(models.iterdir())
    out = tmp_path / "results"
    rc = IC.main(["--models", str(model), "--n", "3", "--n-frame", "1", "--no-gpu", "--out", str(out),
                  "--iterations", str(it), "--ref", "hog_tiny"])
    capsys.readouterr()
    assert rc == 0
    bench = json.loads((out / "costo_inferencia" / "hog_tiny.json").read_text(encoding="utf-8"))
    assert bench["model"]["kind"] == "classical" and bench["gpu_torch"] is None
    assert bench["n_slot"] == 3 and bench["n_frame"] == 1
    df = pd.read_csv(out / "costo_inferencia.csv")
    assert df.loc[0, "cost_per_1000_pred_q"] == pytest.approx(bench["costo_por_1000_pred_Q"], rel=1e-12)
    assert df.loc[0, "inference_cost_month_q"] == pytest.approx(
        PRED_MONTH_COM / 1000 * bench["costo_por_1000_pred_Q"], rel=1e-9)
    assert df.loc[0, "kind"] == "classical" and df.loc[0, "label"] == "B1 (hog_tiny)"
    assert _stamp(models.iterdir()) == models_before
    assert not (models / "benchmark.json").exists()
    assert _protected_stamp() == before


# ===========================================================================
# Tablas versionadas en results/
# ===========================================================================

HAS_RESULTS = ((RESULTS / "costo_inferencia.csv").is_file() and (RESULTS / "iteraciones.csv").is_file()
               and (RESULTS / "costo_inferencia").is_dir())


@pytest.mark.skipif(not HAS_RESULTS, reason="no están results/costo_inferencia.* ni iteraciones.csv")
def test_versioned_tables_rebuild_identically():
    saved = pd.read_csv(RESULTS / "costo_inferencia.csv")
    benches = [json.loads((RESULTS / "costo_inferencia" / f"{n}.json").read_text(encoding="utf-8"))
               for n in saved["name"]]
    df = IC.build_table(benches, RESULTS / "iteraciones.csv", IC.DEFAULT_REF)
    assert list(saved.columns) == list(IC.COLUMNS)
    rebuilt = pd.read_csv(pd.io.common.StringIO(df.to_csv(index=False)))
    pd.testing.assert_frame_equal(rebuilt, saved, check_exact=False, rtol=1e-12)
    md = IC.render_markdown(df, benches, IC.DEFAULT_REF)
    assert md == (RESULTS / "costo_inferencia.md").read_text(encoding="utf-8")
    # Hechos a mano sobre la tabla versionada
    r = saved.set_index("name")
    ref = r.loc[IC.DEFAULT_REF]
    for name, row in r.iterrows():
        assert row["pred_month"] == PRED_MONTH_COM
        assert row["inference_cost_month_q"] == pytest.approx(PRED_MONTH_COM / 1000 * row["cost_per_1000_pred_q"])
        assert row["delta_net_val_q"] == pytest.approx(
            (row["val_value_comercial"] - ref["val_value_comercial"])
            - (row["inference_cost_month_q"] - ref["inference_cost_month_q"]))
    it = pd.read_csv(RESULTS / "iteraciones.csv", encoding="utf-8-sig").set_index("name")
    for name, row in r.iterrows():
        if name in it.index:
            assert row["val_value_comercial"] == pytest.approx(it.loc[name, "val_value_comercial"])
            assert row["test_value_comercial"] == pytest.approx(it.loc[name, "test_value_comercial"])
