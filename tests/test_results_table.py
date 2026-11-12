"""Pruebas de scripts/results_table.py (tabla de iteraciones).

Todo se hace con runs sintéticos en una carpeta temporal (``results_helpers.make_run``):
predicciones de val con semilla fija, sin entrenar ni leer el dataset. Los valores de
referencia se calculan en la prueba de forma independiente (tasas contadas a mano +
``compute_value``). Al final se corre la tabla sobre copias de los runs reales de
``runs/_smoke`` y ``runs/_obsoleto`` (sin modificarlos).
"""

from __future__ import annotations

import getpass
import inspect
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cupo import paths
from cupo.value_model import compute_value, scenario_params, status_quo

from results_helpers import (
    REPO, load_scripts, make_run, manual_rates, metrics_split, scenarios, sigmoid, snapshot,
    synthetic_preds,
)

RT, _ = load_scripts()
SCEN = scenarios()
BOM = b"\xef\xbb\xbf"
# Rutas locales del equipo que no deben aparecer en las salidas
FORBIDDEN = [re.compile(r"[A-Za-z]:[\\/]Us" r"ers[\\/]"), re.compile(r"/ho" r"me/[^/\s]+/")]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def table_for(runs_dir: Path, include_smoke: bool = False) -> pd.DataFrame:
    return RT.build_table(RT.load_records(runs_dir, include_smoke), SCEN)


def row_of(table: pd.DataFrame, group: str) -> pd.Series:
    rows = table[table["group"] == group]
    assert len(rows) == 1, f"se esperaba una fila para {group!r}; hay {len(rows)}"
    return rows.iloc[0]


def independent_value(y, logits, t_L, t_H, cfg, temperature: float = 1.0) -> dict:
    p = sigmoid((logits[:, 1].astype(float) - logits[:, 0].astype(float)) / temperature)
    return compute_value(manual_rates(y, p, t_L, t_H), cfg)


def own_single_threshold(y, p, cfg) -> tuple[float, float]:
    """Mejor t ∈ {0, 0.01, …, 1} con t_L = t_H = t; empate → t más cercano a 0.5."""
    ts = np.round(np.arange(101) / 100, 2)
    vals = np.array([compute_value(manual_rates(y, p, t, t), cfg)["objective"] for t in ts], float)
    best = vals.max()
    tied = np.flatnonzero(vals >= best - 1e-9 * max(1.0, abs(best)))
    k = tied[np.argmin(np.abs(ts[tied] - 0.5))]
    return float(ts[k]), float(vals[k])


def pred_mes(cfg) -> float:
    prm = scenario_params(cfg)
    return prm["espacios"] * (3600.0 / prm["refresh_s"]) * prm["horas_operacion_dia"] * prm["dias_mes"]


def assert_clean_text(text: str, *extra: str) -> None:
    for rx in FORBIDDEN:
        assert not rx.search(text), f"texto prohibido {rx.pattern!r}"
    markers = [str(Path.home()), str(Path.home()).replace("\\", "/"), str(REPO),
               str(REPO).replace("\\", "/"), "C:\\Users", "C:/Users", "/home/", *extra]
    user = getpass.getuser()
    if len(user) >= 3:
        markers.append(user)
    for m in markers:
        assert m not in text, f"ruta o dato local {m!r} en una salida"


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# B0 y valor recalculado
# ---------------------------------------------------------------------------

def test_b0_sintetico_con_valor_cero(runs_dir):
    make_run(runs_dir, "k3_mnv3_ft")
    table = table_for(runs_dir)
    assert table.iloc[0]["iteration"] == "B0"
    b0 = row_of(table, RT.B0_NAME)
    for scen in ("comercial", "uvg"):
        assert b0[f"val_value_{scen}"] == 0.0
        assert b0[f"val_value_{scen}_1thr"] == 0.0
        assert status_quo(SCEN[scen])["objective"] == 0.0
        assert math.isnan(b0[f"delta_value_{scen}"])
    assert b0["val_hours_uvg"] == 0.0
    # sin runs, la tabla solo tiene B0
    empty = RT.build_table([], SCEN)
    assert list(empty["iteration"]) == ["B0"]


def test_valor_recalculado_coincide_con_compute_value(runs_dir):
    run = make_run(runs_dir, "k3_mnv3_ft", sep=1.6, noise=1.3, seed=5)
    row = row_of(table_for(runs_dir), "k3_mnv3_ft")
    for scen, cfg in SCEN.items():
        thr = json.loads((run["run_dir"] / "thresholds.json").read_text(encoding="utf-8"))[scen]
        comp = independent_value(run["y"], run["logits"], thr["t_L"], thr["t_H"], cfg)
        assert row[f"val_value_{scen}_recomputed"] == pytest.approx(comp["objective"], rel=1e-9)
        assert row[f"val_value_{scen}"] == pytest.approx(run["summary"][f"val_value_{scen}"])
        assert row[f"t_L_{scen}"] == thr["t_L"] and row[f"t_H_{scen}"] == thr["t_H"]
        assert row[f"operadores_extra_{scen}"] == pytest.approx(comp["operadores_extra"])
        rates = manual_rates(run["y"], run["p"], thr["t_L"], thr["t_H"])
        assert row[f"val_verify_rate_{scen}"] == pytest.approx(rates["verify_rate"])
        if scen == "uvg":
            assert row["val_hours_uvg"] == pytest.approx(comp["horas_ahorradas_semestre"], rel=1e-9)
    assert not bool(row["value_mismatch"])
    assert "†" not in RT.row_label(row)


@pytest.mark.parametrize("shift", [0.0, -3.0, 3.0])
def test_mejor_umbral_unico_coincide_con_busqueda_propia(runs_dir, shift):
    # shift < 0 empuja las p hacia 0: el mejor umbral único queda por debajo de 0.5.
    y, logits, *_ = synthetic_preds(500, 2.0, 1.0, seed=11)
    p = sigmoid(logits[:, 1] + shift)
    for scen, cfg in SCEN.items():
        t_own, v_own = own_single_threshold(y, p, cfg)
        t, v = RT.best_single_threshold(y, p, cfg)
        assert t == pytest.approx(t_own, abs=1e-9)
        assert v == pytest.approx(v_own, rel=1e-9, abs=1e-6)
        if shift < 0 and scen == "comercial":
            assert t < 0.5, "la búsqueda debe cubrir umbrales menores que 0.5"


def test_mejor_umbral_unico_en_la_tabla(runs_dir):
    run = make_run(runs_dir, "k2_mnv3_head", sep=1.2, noise=1.4)
    row = row_of(table_for(runs_dir), "k2_mnv3_head")
    for scen, cfg in SCEN.items():
        t_own, v_own = own_single_threshold(run["y"], run["p"], cfg)
        assert row[f"t_1thr_{scen}"] == pytest.approx(t_own)
        assert row[f"val_value_{scen}_1thr"] == pytest.approx(v_own, rel=1e-9, abs=1e-6)


# ---------------------------------------------------------------------------
# Agrupación por name y semillas
# ---------------------------------------------------------------------------

def test_agrupa_por_name_con_media_y_desviacion(runs_dir, tmp_path):
    runs = [make_run(runs_dir, "k6_mnv3_ft_seeds", seed=s, sep=1.5 + 0.2 * s, train_time_s=60.0 * s)
            for s in (1, 2, 3)]
    make_run(runs_dir, "k3_mnv3_ft", seed=7, summary_extra={"seed": 7.0})
    table = table_for(runs_dir)
    row = row_of(table, "k6_mnv3_ft_seeds")
    assert row["n_runs"] == 3
    assert row["seeds"] == "1, 2, 3"
    assert row_of(table, "k3_mnv3_ft")["seeds"] == "7"  # semilla float en el JSON → entero
    for col in ("val_f1_macro", "val_value_comercial", "val_value_uvg"):
        vals = np.array([r["summary"][col] for r in runs], float)
        assert row[col] == pytest.approx(vals.mean())
        assert row[f"{col}_std"] == pytest.approx(vals.std(ddof=1))
    assert row["train_time_s_total"] == pytest.approx(60.0 + 120.0 + 180.0)
    assert math.isnan(row_of(table, "k3_mnv3_ft")["val_f1_macro_std"])

    out = tmp_path / "results"
    RT.generate(runs_dir, out)
    md = (out / "iteraciones.md").read_text(encoding="utf-8")
    line = next(l for l in md.splitlines() if "(k6_mnv3_ft_seeds)" in l and "Validación" not in l)
    assert "±" in line and "1, 2, 3" in line
    k3_line = next(l for l in md.splitlines() if "(k3_mnv3_ft)" in l)
    assert "7.0" not in k3_line.split("|")[4]
    csv = pd.read_csv(out / "iteraciones.csv", encoding="utf-8-sig", dtype={"seeds": str})
    assert csv.loc[csv["group"] == "k6_mnv3_ft_seeds", "seeds"].iloc[0] == "1, 2, 3"
    assert csv.loc[csv["group"] == "k3_mnv3_ft", "seeds"].iloc[0] == "7"


# ---------------------------------------------------------------------------
# Subconjuntos: no compiten; Δ dentro del mismo subset
# ---------------------------------------------------------------------------

def test_runs_con_subset_no_compiten(runs_dir, tmp_path):
    make_run(runs_dir, "k1_malexnet_aug", sep=1.0, noise=1.5)
    make_run(runs_dir, "k2_mnv3_head", sep=1.4, noise=1.3)
    # run de humo con un valor guardado enorme: no debe ganar nada
    make_run(runs_dir, "k3_mnv3_ft", subset=0.02, stored_values={"comercial": 1e9, "uvg": 1e9})
    records = RT.load_records(runs_dir)
    best = RT.best_record(records)
    full = [r for r in records if r.is_full]
    assert best.name == max(full, key=RT.record_value).name
    assert best.subset == 1.0
    reps = RT.representatives(records)
    assert {r.group for r in reps} == {"k1_malexnet_aug", "k2_mnv3_head"}

    table = RT.build_table(records, SCEN)
    assert "k3_mnv3_ft (subset 0.02)" in set(table["group"])
    smoke_row = row_of(table, "k3_mnv3_ft (subset 0.02)")
    assert smoke_row["delta_ref_comercial"] == "B0"
    assert smoke_row["delta_value_comercial"] == pytest.approx(1e9)
    k2 = row_of(table, "k2_mnv3_head")
    k1 = row_of(table, "k1_malexnet_aug")
    assert k2["delta_ref_comercial"] == "K1"
    assert k2["delta_value_comercial"] == pytest.approx(k2["val_value_comercial"] - k1["val_value_comercial"])

    summary = RT.summary_rows(table)
    assert "K3" not in set(summary["iteration"])
    assert (summary["subset"].astype(float) >= 1).all()

    rankings = RT.compute_rankings(records, SCEN)
    for rank in rankings.values():
        assert "k3_mnv3_ft (subset 0.02)" not in rank.columns
        assert set(rank["winner"]) <= {"k1_malexnet_aug", "k2_mnv3_head"}

    out = tmp_path / "results"
    RT.generate(runs_dir, out)
    md = (out / "iteraciones.md").read_text(encoding="utf-8")
    assert "Atención" in md
    resumen = (out / "iteraciones_resumen.md").read_text(encoding="utf-8")
    assert "subset" not in resumen and "**K3**" not in resumen


def test_smoke_solo_con_include_smoke(runs_dir):
    make_run(runs_dir, "k2_mnv3_head")
    make_run(runs_dir / "_smoke", "k3_mnv3_ft", subset=0.02)
    assert {r.name for r in RT.load_records(runs_dir)} == {"k2_mnv3_head"}
    assert {r.name for r in RT.load_records(runs_dir, include_smoke=True)} == {"k2_mnv3_head", "k3_mnv3_ft"}


def test_solo_subconjuntos_avisa_y_no_falla(runs_dir, tmp_path, capsys):
    make_run(runs_dir / "_smoke", "k1_malexnet_aug", subset=0.02, sep=1.0)
    make_run(runs_dir / "_smoke", "k2_mnv3_head", subset=0.02, sep=2.0)
    records = RT.load_records(runs_dir, include_smoke=True)
    assert RT.best_record(records) is not None
    RT.generate(runs_dir, tmp_path / "results", include_smoke=True)
    assert "no son concluyentes" in capsys.readouterr().out
    md = (tmp_path / "results" / "iteraciones.md").read_text(encoding="utf-8")
    assert "no es concluyente" in md


def test_delta_dentro_del_mismo_subset_valores_a_mano():
    table = pd.DataFrame([
        {"iteration": "B0", "group": "b0", "subset": 1.0, "val_value_comercial": 0.0},
        {"iteration": "K1", "group": "k1", "subset": 1.0, "val_value_comercial": 100.0},
        {"iteration": "K1", "group": "k1 (subset 0.02)", "subset": 0.02, "val_value_comercial": 1000.0},
        {"iteration": "K3", "group": "k3a", "subset": 1.0, "val_value_comercial": 150.0},
        {"iteration": "K3", "group": "k3b", "subset": 1.0, "val_value_comercial": 300.0},
        {"iteration": "K3", "group": "k3 (subset 0.02)", "subset": 0.02, "val_value_comercial": 1500.0},
        {"iteration": "K4", "group": "k4", "subset": 1.0, "val_value_comercial": 250.0},
        {"iteration": "K5", "group": "k5 (subset 0.05)", "subset": 0.05, "val_value_comercial": 70.0},
    ])
    out = RT.add_deltas(RT.sort_table(table), scenarios=("comercial",)).set_index("group")
    d, ref = out["delta_value_comercial"], out["delta_ref_comercial"]
    assert math.isnan(d["b0"]) and ref["b0"] == ""
    assert d["k1"] == 100.0 and ref["k1"] == "B0"
    assert d["k1 (subset 0.02)"] == 1000.0 and ref["k1 (subset 0.02)"] == "B0"
    assert d["k3a"] == 50.0 and ref["k3a"] == "K1"
    assert d["k3b"] == 200.0
    assert d["k3 (subset 0.02)"] == 500.0 and ref["k3 (subset 0.02)"] == "K1"
    # K4 contra la mejor fila completa de K3 (k3b = 300), no contra el humo (1500)
    assert d["k4"] == -50.0 and ref["k4"] == "K3 (k3b)"
    # un subset sin filas previas del mismo subset se compara con B0
    assert d["k5 (subset 0.05)"] == 70.0 and ref["k5 (subset 0.05)"] == "B0"


def test_delta_integracion_mismo_subset(runs_dir):
    make_run(runs_dir, "k1_malexnet_aug", stored_values={"comercial": 1000.0})
    make_run(runs_dir, "k2_mnv3_head", stored_values={"comercial": 1300.0})
    make_run(runs_dir, "k1_malexnet_aug", seed=1, subset=0.02, stored_values={"comercial": 5000.0})
    make_run(runs_dir, "k2_mnv3_head", seed=1, subset=0.02, stored_values={"comercial": 5400.0})
    table = table_for(runs_dir)
    assert row_of(table, "k2_mnv3_head")["delta_value_comercial"] == pytest.approx(300.0)
    assert row_of(table, "k2_mnv3_head (subset 0.02)")["delta_value_comercial"] == pytest.approx(400.0)
    assert row_of(table, "k1_malexnet_aug (subset 0.02)")["delta_value_comercial"] == pytest.approx(5000.0)


# ---------------------------------------------------------------------------
# Marca †, calibración, test/OOD, costo de inferencia
# ---------------------------------------------------------------------------

def test_marca_cuando_el_valor_guardado_no_coincide(runs_dir, tmp_path, capsys):
    ok = make_run(runs_dir, "k1_malexnet_aug", sep=1.5)
    bad = make_run(runs_dir, "k2_mnv3_head", sep=1.8, seed=3)
    true_value = bad["values"]["comercial"]
    summary_path = bad["run_dir"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["val_value_comercial"] = true_value + 500.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    table = table_for(runs_dir)
    assert "no coinciden" in capsys.readouterr().out
    bad_row, ok_row = row_of(table, "k2_mnv3_head"), row_of(table, "k1_malexnet_aug")
    assert bool(bad_row["value_mismatch"]) and not bool(ok_row["value_mismatch"])
    assert bad_row["val_value_comercial"] == pytest.approx(true_value + 500.0)  # se muestra el guardado
    assert bad_row["val_value_comercial_recomputed"] == pytest.approx(true_value, rel=1e-9)
    assert RT.row_label(bad_row).endswith("†")
    assert "†" not in RT.row_label(ok_row)
    assert ok["values"]["comercial"] == pytest.approx(ok_row["val_value_comercial"])

    out = tmp_path / "results"
    RT.generate(runs_dir, out)
    md = (out / "iteraciones.md").read_text(encoding="utf-8")
    assert "(k2_mnv3_head) †" in md and "(k1_malexnet_aug) †" not in md
    assert "**†**" in md
    csv = pd.read_csv(out / "iteraciones.csv", encoding="utf-8-sig")
    assert csv.set_index("group").loc["k2_mnv3_head", "value_mismatch"] in (True, "True")


def test_con_calibracion_usa_temperatura_y_sus_umbrales(runs_dir, tmp_path):
    T, seed, sep, noise, n = 2.0, 9, 1.4, 1.5, 400
    y, logits, *_ = synthetic_preds(n, sep, noise, seed)
    thr = {"comercial": {"t_L": 0.3, "t_H": 0.7}, "uvg": {"t_L": 0.2, "t_H": 0.6}}
    vals = {s: float(independent_value(y, logits, thr[s]["t_L"], thr[s]["t_H"], SCEN[s], T)["objective"])
            for s in SCEN}
    calib = {"temperature": T, "fitted_on": "val", "thresholds": thr,
             "value_val": {s: {"objective_value": vals[s], "ci95": [vals[s] - 5, vals[s] + 5]} for s in SCEN}}
    run = make_run(runs_dir, "k6_mnv3_ft_seeds", seed=seed, sep=sep, noise=noise, n=n, calibration=calib)
    row = row_of(table_for(runs_dir), "k6_mnv3_ft_seeds")
    assert row["temperature"] == pytest.approx(T)
    assert row["calibrated"] == "sí"
    for s in SCEN:
        assert row[f"t_L_{s}"] == thr[s]["t_L"] and row[f"t_H_{s}"] == thr[s]["t_H"]
        assert row[f"t_L_{s}"] != run["thresholds"][s]["t_L"] or row[f"t_H_{s}"] != run["thresholds"][s]["t_H"]
        assert row[f"val_value_{s}"] == pytest.approx(vals[s])
        assert row[f"val_value_{s}"] != pytest.approx(run["summary"][f"val_value_{s}"])
        assert row[f"val_value_{s}_ci_lo"] == pytest.approx(vals[s] - 5)
        assert row[f"val_value_{s}_recomputed"] == pytest.approx(vals[s], rel=1e-9)
        # el mejor umbral único también usa las p calibradas
        p_cal = sigmoid(logits[:, 1] / T)
        assert row[f"val_value_{s}_1thr"] == pytest.approx(own_single_threshold(y, p_cal, SCEN[s])[1], rel=1e-9)
    assert not bool(row["value_mismatch"])
    out = tmp_path / "results"
    RT.generate(runs_dir, out)
    assert "sí (T = 2.000)" in (out / "iteraciones.md").read_text(encoding="utf-8")


def test_columnas_test_y_ood_solo_si_existen(runs_dir, tmp_path):
    a = make_run(runs_dir, "k2_mnv3_head")
    out = tmp_path / "r0"
    RT.generate(runs_dir, out)
    cols = pd.read_csv(out / "iteraciones.csv", encoding="utf-8-sig").columns
    assert not any(c.startswith(("test_", "ood_")) for c in cols)
    md = (out / "iteraciones.md").read_text(encoding="utf-8")
    assert "## Test" not in md and "OOD" not in md

    (a["run_dir"] / "metrics_test.json").write_text(json.dumps(metrics_split(0.97, 0.01, 12345.0)),
                                                    encoding="utf-8")
    out = tmp_path / "r1"
    RT.generate(runs_dir, out)
    csv = pd.read_csv(out / "iteraciones.csv", encoding="utf-8-sig")
    assert {"test_f1_macro", "test_rate_busy_as_free", "test_value_comercial"} <= set(csv.columns)
    assert not any(c.startswith("ood_") for c in csv.columns)
    assert csv.set_index("group").loc["k2_mnv3_head", "test_value_comercial"] == pytest.approx(12345.0)
    md = (out / "iteraciones.md").read_text(encoding="utf-8")
    assert "## Test" in md and "Test F1 macro" in md and "OOD (CNRPark" not in md

    (a["run_dir"] / "metrics_ood.json").write_text(json.dumps(metrics_split(0.9, 0.05, 999.0)),
                                                   encoding="utf-8")
    out = tmp_path / "r2"
    RT.generate(runs_dir, out)
    csv = pd.read_csv(out / "iteraciones.csv", encoding="utf-8-sig")
    assert {"ood_f1_macro", "ood_rate_busy_as_free", "ood_value_comercial"} <= set(csv.columns)
    assert "OOD (CNRPark A/B) F1 macro" in (out / "iteraciones.md").read_text(encoding="utf-8")


def test_costo_de_inferencia_desde_benchmark_o_costs(runs_dir, tmp_path):
    bench = {"cpu_1_thread": {"per_slot": {"p50_ms": 3.25, "p95_ms": 4.0}},
             "costo_por_1000_pred_Q": 0.5}
    make_run(runs_dir, "k2_mnv3_head", benchmark=bench)
    make_run(runs_dir, "k3_mnv3_ft")
    table = table_for(runs_dir)
    with_b, without = row_of(table, "k2_mnv3_head"), row_of(table, "k3_mnv3_ft")
    n_pred = pred_mes(SCEN["comercial"])
    assert n_pred == pytest.approx(164 * 120 * 14 * 30)  # 8,265,600 con los YAML actuales
    assert with_b["inference_cost_month_q"] == pytest.approx(n_pred / 1000 * 0.5)
    assert with_b["inference_cost_source"] == "benchmark.json"
    assert with_b["latency_ms"] == pytest.approx(3.25)
    assert with_b["cost_per_1000_pred_q"] == pytest.approx(0.5)
    default = yaml.safe_load((REPO / "configs" / "costs.yaml").read_text(encoding="utf-8"))["costo_por_1000_pred_Q"]
    assert without["inference_cost_month_q"] == pytest.approx(n_pred / 1000 * default)
    assert without["inference_cost_source"] == "costs.yaml"
    assert math.isnan(without["latency_ms"])

    out = tmp_path / "results"
    RT.generate(runs_dir, out)
    md = (out / "iteraciones.md").read_text(encoding="utf-8")
    assert "(benchmark.json)" in md and "(costs.yaml)" in md


def test_latencia_con_esquema_alternativo():
    assert RT.latency_ms({"ort_cpu": {"batch1": {"p50_ms": 2.0}}, "gpu": {"p50_ms": 0.1}}) == pytest.approx(2.0)
    assert math.isnan(RT.latency_ms(None)) and math.isnan(RT.latency_ms({"x": 1}))


# ---------------------------------------------------------------------------
# Archivos de salida
# ---------------------------------------------------------------------------

def test_salidas_csv_utf8_sig_y_resumen(runs_dir, tmp_path):
    make_run(runs_dir, "b2_malexnet", sep=1.0, noise=1.6)
    make_run(runs_dir, "k3_mnv3_ft", sep=2.0)
    out = tmp_path / "results"
    written = RT.generate(runs_dir, out)
    names = {p.name for p in written}
    assert {"iteraciones.csv", "iteraciones.md", "iteraciones_resumen.md",
            "ranking_sensibilidad_comercial.csv", "ranking_sensibilidad_uvg.csv"} <= names
    for p in written:
        assert p.parent == out and p.is_file()
        if p.suffix == ".csv":
            assert p.read_bytes().startswith(BOM), p.name
    csv = pd.read_csv(out / "iteraciones.csv", encoding="utf-8-sig")
    assert csv.columns[0] == "iteration"
    assert list(csv["iteration"]) == ["B0", "B2", "K3"]
    resumen = (out / "iteraciones_resumen.md").read_text(encoding="utf-8")
    assert resumen.startswith("# Resumen de iteraciones")
    assert all(f"**{it}**" in resumen for it in ("B0", "B2", "K3"))


def test_cli_escribe_en_out(runs_dir, tmp_path, monkeypatch):
    make_run(runs_dir, "k3_mnv3_ft")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    before = snapshot(paths.results_dir())
    assert RT.main(["--runs-dir", str(runs_dir), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "iteraciones.md").is_file()
    assert list(cwd.iterdir()) == []
    assert snapshot(paths.results_dir()) == before


# ---------------------------------------------------------------------------
# Robustez
# ---------------------------------------------------------------------------

def test_summary_corrupto_no_detiene_la_tabla(runs_dir, tmp_path, capsys):
    make_run(runs_dir, "k2_mnv3_head")
    bad = make_run(runs_dir, "k3_mnv3_ft")
    (bad["run_dir"] / "summary.json").write_text("{esto no es json", encoding="utf-8")
    (runs_dir / "k4_sin_summary").mkdir()
    out = tmp_path / "results"
    RT.generate(runs_dir, out)
    log = capsys.readouterr().out
    assert "Aviso" in log and bad["run_id"] in log and "k4_sin_summary" in log
    csv = pd.read_csv(out / "iteraciones.csv", encoding="utf-8-sig")
    assert set(csv["group"]) == {RT.B0_NAME, "k2_mnv3_head"}


def test_history_vacio_no_afecta_la_tabla(runs_dir, tmp_path):
    run = make_run(runs_dir, "k2_mnv3_head")
    (run["run_dir"] / "history.csv").write_text("", encoding="utf-8")
    RT.generate(runs_dir, tmp_path / "results")
    assert (tmp_path / "results" / "iteraciones.md").is_file()


def test_calibracion_sin_value_val_usa_el_recalculado(runs_dir, capsys):
    T = 1.5
    y, logits, *_ = synthetic_preds(400, 2.0, 1.2, 42)
    thr = {"comercial": {"t_L": 0.25, "t_H": 0.75}, "uvg": {"t_L": 0.25, "t_H": 0.75}}
    make_run(runs_dir, "k6_mnv3_ft_seeds", calibration={"temperature": T, "thresholds": thr})
    row = row_of(table_for(runs_dir), "k6_mnv3_ft_seeds")
    assert "no tiene valor guardado" in capsys.readouterr().out
    expected = independent_value(y, logits, 0.25, 0.75, SCEN["comercial"], T)["objective"]
    assert row["val_value_comercial"] == pytest.approx(expected, rel=1e-9)
    assert math.isnan(row["val_value_comercial_ci_lo"])
    assert not bool(row["value_mismatch"])


def test_metrics_test_corrupto_se_ignora(runs_dir, capsys):
    run = make_run(runs_dir, "k2_mnv3_head")
    (run["run_dir"] / "metrics_test.json").write_text("[[[", encoding="utf-8")
    (run["run_dir"] / "benchmark.json").write_text("nope", encoding="utf-8")
    table = table_for(runs_dir)
    assert "Aviso" in capsys.readouterr().out
    assert "test_f1_macro" not in table.columns


# ---------------------------------------------------------------------------
# Ranking de sensibilidad
# ---------------------------------------------------------------------------

def test_ranking_sensibilidad_columnas_y_coherencia(runs_dir, tmp_path):
    runs = {
        "b2_malexnet": make_run(runs_dir, "b2_malexnet", sep=1.0, noise=1.5, seed=1),
        "k2_mnv3_head": make_run(runs_dir, "k2_mnv3_head", sep=1.6, noise=1.3, seed=2),
        "k3_mnv3_ft": make_run(runs_dir, "k3_mnv3_ft", sep=1.7, noise=1.2, seed=3),
    }
    out = tmp_path / "results"
    RT.generate(runs_dir, out)
    groups = list(runs)
    for scen, cfg in SCEN.items():
        rank = pd.read_csv(out / f"ranking_sensibilidad_{scen}.csv", encoding="utf-8-sig")
        expected = {"param", "factor", "param_value", "clipped", "winner", "runner_up", "margin",
                    "base_winner", "winner_changed", "order_changed", *groups}
        assert expected <= set(rank.columns)
        assert set(rank["factor"]) == {0.5, 0.8, 1.2, 1.5}
        # ganador en el caso base: valor independiente en el punto de operación de cada run
        base = {g: compute_value(manual_rates(r["y"], r["p"], r["thresholds"][scen]["t_L"],
                                              r["thresholds"][scen]["t_H"]), cfg)["objective"]
                for g, r in runs.items()}
        base_order = sorted(base, key=base.get, reverse=True)
        assert set(rank["base_winner"]) == {base_order[0]}
        for _, r in rank.iterrows():
            vals = r[groups].astype(float)
            order = list(vals.sort_values(ascending=False).index)
            assert r["winner"] == order[0]
            assert r["runner_up"] == order[1]
            assert r["margin"] == pytest.approx(vals[order[0]] - vals[order[1]])
            assert r["margin"] >= 0
            assert bool(r["winner_changed"]) == (r["winner"] != r["base_winner"])
            assert bool(r["order_changed"]) == (order != base_order)
            if r["winner_changed"]:
                assert r["order_changed"]
    md = (out / "iteraciones.md").read_text(encoding="utf-8")
    assert "¿Cambia el ranking con los supuestos?" in md


def test_ranking_valores_de_cada_fila_coinciden_con_compute_value(runs_dir, tmp_path):
    a = make_run(runs_dir, "k1_malexnet_aug", sep=1.0, seed=1)
    b = make_run(runs_dir, "k2_mnv3_head", sep=2.0, seed=2)
    reps = RT.representatives(RT.load_records(runs_dir))
    rank = RT.ranking_sensitivity(reps, "comercial", SCEN["comercial"])
    from cupo.value_model import with_param

    row = rank[(rank["param"] == "ocupacion") & (rank["factor"] == 0.8)].iloc[0]
    cfg = with_param(SCEN["comercial"], "ocupacion", row["param_value"])
    for name, run in (("k1_malexnet_aug", a), ("k2_mnv3_head", b)):
        thr = run["thresholds"]["comercial"]
        exp = compute_value(manual_rates(run["y"], run["p"], thr["t_L"], thr["t_H"]), cfg)["objective"]
        assert row[name] == pytest.approx(exp, rel=1e-9)


def test_ranking_con_un_solo_modelo_no_escribe_csv(runs_dir, tmp_path):
    make_run(runs_dir, "k2_mnv3_head")
    written = RT.generate(runs_dir, tmp_path / "results")
    assert not any(p.name.startswith("ranking_") for p in written)
    md = (tmp_path / "results" / "iteraciones.md").read_text(encoding="utf-8")
    assert "al menos 2 modelos" in md


# ---------------------------------------------------------------------------
# Invariante: test nunca elige el mejor run
# ---------------------------------------------------------------------------

def test_test_no_cambia_el_mejor_run(runs_dir, tmp_path):
    good = make_run(runs_dir, "k3_mnv3_ft", sep=2.5, noise=1.0, seed=1)
    bad = make_run(runs_dir, "k3_mnv3_ft_lr", sep=0.8, noise=1.5, seed=2)
    assert good["values"]["comercial"] > bad["values"]["comercial"]
    rec0 = RT.load_records(runs_dir)
    best0 = RT.best_record(rec0).run_id
    reps0 = [r.run_id for r in RT.representatives(rec0)]
    summary0 = list(RT.summary_rows(RT.build_table(rec0, SCEN))["group"])

    (bad["run_dir"] / "metrics_test.json").write_text(
        json.dumps(metrics_split(0.9999, 0.0, 1e9)), encoding="utf-8")
    (bad["run_dir"] / "metrics_ood.json").write_text(
        json.dumps(metrics_split(0.9999, 0.0, 1e9)), encoding="utf-8")
    rec1 = RT.load_records(runs_dir)
    assert RT.best_record(rec1).run_id == best0 == good["run_id"]
    assert [r.run_id for r in RT.representatives(rec1)] == reps0
    table1 = RT.build_table(rec1, SCEN)
    assert list(RT.summary_rows(table1)["group"]) == summary0
    assert "k3_mnv3_ft" in summary0 and "k3_mnv3_ft_lr" not in summary0
    resumen = RT.render_summary(table1)
    assert "(k3_mnv3_ft)" in resumen and "(k3_mnv3_ft_lr)" not in resumen


def test_funciones_de_seleccion_no_leen_test():
    for fn in (RT.best_record, RT.representatives, RT.record_value, RT.source_value,
               RT.stored_value, RT.comparable_records, RT.summary_rows, RT.compute_rankings,
               RT.ranking_sensitivity, RT.operating_rates):
        src = inspect.getsource(fn)
        assert "test" not in src and "ood" not in src.lower(), fn.__name__
    src = (REPO / "scripts" / "results_table.py").read_text(encoding="utf-8")
    assert "cupo.data" not in src and "load_split" not in src and "allow_test" not in src


# ---------------------------------------------------------------------------
# Texto generado
# ---------------------------------------------------------------------------

def test_textos_sin_menciones_prohibidas_ni_rutas_locales(runs_dir, tmp_path):
    make_run(runs_dir, "b2_malexnet", sep=1.0, benchmark={"costo_por_1000_pred_Q": 0.1})
    make_run(runs_dir, "k3_mnv3_ft", sep=2.0, metrics_test=metrics_split(0.95, 0.02, 100.0))
    make_run(runs_dir, "k6_mnv3_ft_seeds", seed=1)
    make_run(runs_dir, "k6_mnv3_ft_seeds", seed=2)
    out = tmp_path / "results"
    written = RT.generate(runs_dir, out)
    for p in written:
        assert_clean_text(p.read_text(encoding="utf-8-sig"), str(tmp_path), str(tmp_path).replace("\\", "/"))
    for script in ("results_table.py", "make_figures.py"):
        src = (REPO / "scripts" / script).read_text(encoding="utf-8")
        for rx in FORBIDDEN:
            assert not rx.search(src), f"{script}: {rx.pattern}"


# ---------------------------------------------------------------------------
# Runs reales (copias de runs/_smoke y runs/_obsoleto)
# ---------------------------------------------------------------------------

REAL_RUNS = paths.runs_dir()


@pytest.mark.skipif(not (REAL_RUNS / "_smoke").is_dir(), reason="no hay runs/_smoke")
def test_runs_reales_copiados(tmp_path, capsys):
    root = tmp_path / "runs"
    before = snapshot(REAL_RUNS)
    shutil.copytree(REAL_RUNS / "_smoke", root / "_smoke")
    obsolete = REAL_RUNS / "_obsoleto"
    if obsolete.is_dir():
        for d in obsolete.iterdir():
            if d.is_dir():
                shutil.copytree(d, root / d.name)
    out = tmp_path / "results"
    written = RT.generate(root, out, include_smoke=True)
    assert snapshot(REAL_RUNS) == before  # los originales no se tocan
    names = {p.name for p in written}
    assert {"iteraciones.csv", "iteraciones.md", "iteraciones_resumen.md"} <= names
    csv = pd.read_csv(out / "iteraciones.csv", encoding="utf-8-sig")
    records = RT.load_records(root, include_smoke=True)
    assert len(records) >= 1
    full = [r for r in records if r.is_full]
    best = RT.best_record(records)
    if full and any(r.val is not None for r in full):
        assert best.is_full
    assert csv["iteration"].iloc[0] == "B0"
    for p in written:
        assert_clean_text(p.read_text(encoding="utf-8-sig"), str(tmp_path))
