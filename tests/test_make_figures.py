"""Pruebas de scripts/make_figures.py (figuras de resultados).

Runs sintéticos en una carpeta temporal (``results_helpers.make_run``); nada se entrena ni se
lee del dataset. Se verifica el conjunto de figuras, que sean PNG válidos, que una figura
rota no frene las demás, que no se escriba nada fuera de ``--out``/``--table-out`` y que el
mejor run se elija solo con val.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cupo import paths

from results_helpers import (
    REPO, load_scripts, make_run, metrics_split, sigmoid, snapshot, synthetic_preds,
)

RT, MF = load_scripts()
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_VAL = {
    "curvas_entrenamiento.png", "valor_vs_umbral.png", "valor_vs_metricas.png",
    "tornado_comercial.png", "tornado_uvg.png", "valor_vs_costo.png",
    "matrices_confusion_val.png", "matrices_tres_estados_val.png",
    "metricas_por_camara_val.png", "metricas_por_clima_val.png",
}


def split_figures(split: str) -> set[str]:
    return {f"matrices_confusion_{split}.png", f"matrices_tres_estados_{split}.png",
            f"metricas_por_camara_{split}.png", f"metricas_por_clima_{split}.png"}


def pngs(folder: Path) -> set[str]:
    return {p.name for p in folder.glob("*.png")}


def assert_valid_png(path: Path) -> None:
    assert path.read_bytes()[:8] == PNG_SIGNATURE, path.name
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        assert im.format == "PNG" and im.size[0] > 100 and im.size[1] > 100


def three_runs(runs_dir: Path, **extra) -> dict[str, dict]:
    return {
        "b2_malexnet": make_run(runs_dir, "b2_malexnet", sep=1.0, noise=1.5, seed=1,
                                params_total=40_000, train_time_s=50.0),
        "k2_mnv3_head": make_run(runs_dir, "k2_mnv3_head", sep=1.6, noise=1.3, seed=2,
                                 params_total=1_000_000, train_time_s=200.0,
                                 benchmark={"cpu_1_thread": {"per_slot": {"p50_ms": 2.0}},
                                            "costo_por_1000_pred_Q": 0.01}),
        "k3_mnv3_ft": make_run(runs_dir, "k3_mnv3_ft", sep=2.2, noise=1.1, seed=3,
                               params_total=1_500_000, train_time_s=900.0, **extra),
    }


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture()
def cwd(tmp_path: Path, monkeypatch) -> Path:
    d = tmp_path / "cwd"
    d.mkdir()
    monkeypatch.chdir(d)
    return d


# ---------------------------------------------------------------------------
# Conjunto de figuras
# ---------------------------------------------------------------------------

def test_genera_las_diez_figuras_validas(runs_dir, tmp_path, cwd):
    three_runs(runs_dir)
    out = tmp_path / "figs"
    assert MF.main(["--runs-dir", str(runs_dir), "--out", str(out)]) == 0
    assert pngs(out) == EXPECTED_VAL
    for name in EXPECTED_VAL:
        assert_valid_png(out / name)
    for scen in ("comercial", "uvg"):
        csv = out / f"tornado_{scen}.csv"
        assert csv.read_bytes().startswith(b"\xef\xbb\xbf")
        df = pd.read_csv(csv, encoding="utf-8-sig")
        assert {"param", "factor", "delta", "param_label"} <= set(df.columns)
    assert list(cwd.iterdir()) == []


def test_figuras_de_test_y_ood_solo_si_existen(runs_dir, tmp_path, cwd):
    runs = three_runs(runs_dir, extra_split_preds=("test",))
    out = tmp_path / "figs_test"
    assert MF.main(["--runs-dir", str(runs_dir), "--out", str(out)]) == 0
    assert pngs(out) == EXPECTED_VAL | split_figures("test")

    ys, lg, cam, wea, dt, fl = synthetic_preds(150, 1.5, 1.0, 77)
    np.savez_compressed(runs["k2_mnv3_head"]["run_dir"] / "ood_preds.npz", y=ys,
                        p=sigmoid(lg[:, 1]), logits=lg, camera=cam, weather=wea, date=dt, path=fl)
    out = tmp_path / "figs_all"
    assert MF.main(["--runs-dir", str(runs_dir), "--out", str(out)]) == 0
    assert pngs(out) == EXPECTED_VAL | split_figures("test") | split_figures("ood")
    for name in pngs(out):
        assert_valid_png(out / name)


def test_with_table_tambien_escribe_la_tabla(runs_dir, tmp_path, cwd):
    three_runs(runs_dir)
    out, table_out = tmp_path / "figs", tmp_path / "tabla"
    rc = MF.main(["--runs-dir", str(runs_dir), "--out", str(out), "--with-table",
                  "--table-out", str(table_out)])
    assert rc == 0
    assert pngs(out) == EXPECTED_VAL
    tables = {p.name for p in table_out.iterdir()}
    assert {"iteraciones.csv", "iteraciones.md", "iteraciones_resumen.md",
            "ranking_sensibilidad_comercial.csv", "ranking_sensibilidad_uvg.csv"} <= tables
    assert not any(n.endswith(".png") for n in tables)
    assert not any(p.name.startswith("iteraciones") for p in out.iterdir())


def test_sin_with_table_no_escribe_tabla(runs_dir, tmp_path, cwd):
    three_runs(runs_dir)
    before = snapshot(paths.results_dir())
    out = tmp_path / "figs"
    assert MF.main(["--runs-dir", str(runs_dir), "--out", str(out)]) == 0
    assert not any(p.suffix == ".md" for p in out.iterdir())
    assert snapshot(paths.results_dir()) == before


def test_una_figura_que_falla_no_detiene_las_demas(runs_dir, tmp_path, cwd, monkeypatch, capsys):
    three_runs(runs_dir)

    def boom(*args, **kwargs):
        raise RuntimeError("falla simulada")

    monkeypatch.setattr(MF, "fig_value_vs_metrics", boom)
    monkeypatch.setattr(MF, "fig_tornado", boom)
    out = tmp_path / "figs"
    assert MF.main(["--runs-dir", str(runs_dir), "--out", str(out)]) == 0
    log = capsys.readouterr().out
    assert "no se pudo generar valor frente a métricas" in log
    assert "no se pudo generar tornado comercial" in log and "no se pudo generar tornado uvg" in log
    assert pngs(out) == EXPECTED_VAL - {"valor_vs_metricas.png", "tornado_comercial.png",
                                         "tornado_uvg.png"}


def test_nada_se_escribe_fuera_de_out_y_table_out(runs_dir, tmp_path, cwd):
    three_runs(runs_dir, extra_split_preds=("test",))
    runs_before = snapshot(runs_dir)
    results_before = snapshot(paths.results_dir())
    repo_top_before = sorted(p.name for p in REPO.iterdir())
    out, table_out = tmp_path / "salida" / "figs", tmp_path / "salida" / "tabla"
    assert MF.main(["--runs-dir", str(runs_dir), "--out", str(out), "--with-table",
                    "--table-out", str(table_out)]) == 0
    assert snapshot(runs_dir) == runs_before
    assert snapshot(paths.results_dir()) == results_before
    assert sorted(p.name for p in REPO.iterdir()) == repo_top_before
    assert list(cwd.iterdir()) == []
    top = {p.name for p in tmp_path.iterdir()}
    assert top == {"runs", "cwd", "salida"}
    assert {p.name for p in (tmp_path / "salida").iterdir()} == {"figs", "tabla"}


def test_sin_runs_devuelve_error_y_no_genera(tmp_path, cwd, capsys):
    empty = tmp_path / "runs"
    empty.mkdir()
    out = tmp_path / "figs"
    assert MF.main(["--runs-dir", str(empty), "--out", str(out)]) == 1
    assert not out.exists() or not any(out.iterdir())


# ---------------------------------------------------------------------------
# Robustez
# ---------------------------------------------------------------------------

def test_history_vacio_y_summary_corrupto_no_detienen(runs_dir, tmp_path, cwd, capsys):
    runs = three_runs(runs_dir)
    (runs["b2_malexnet"]["run_dir"] / "history.csv").write_text("", encoding="utf-8")
    bad = make_run(runs_dir, "k4_mnv3_ft_cost", seed=4)
    (bad["run_dir"] / "summary.json").write_text("{roto", encoding="utf-8")
    out = tmp_path / "figs"
    assert MF.main(["--runs-dir", str(runs_dir), "--out", str(out), "--with-table",
                    "--table-out", str(tmp_path / "tabla")]) == 0
    log = capsys.readouterr().out
    assert "history.csv" in log and "Aviso" in log
    assert bad["run_id"] in log
    assert pngs(out) == EXPECTED_VAL
    assert (tmp_path / "tabla" / "iteraciones.md").is_file()


def test_history_sin_columnas_y_calibracion_sin_value_val(runs_dir, tmp_path, cwd, capsys):
    runs = three_runs(runs_dir)
    (runs["k2_mnv3_head"]["run_dir"] / "history.csv").write_text("epoch,foo\n1,2\n", encoding="utf-8")
    thr = {"comercial": {"t_L": 0.3, "t_H": 0.7}, "uvg": {"t_L": 0.3, "t_H": 0.7}}
    make_run(runs_dir, "k6_mnv3_ft_seeds", seed=6, sep=2.0,
             calibration={"temperature": 1.3, "thresholds": thr})
    out = tmp_path / "figs"
    assert MF.main(["--runs-dir", str(runs_dir), "--out", str(out), "--with-table",
                    "--table-out", str(tmp_path / "tabla")]) == 0
    log = capsys.readouterr().out
    assert "sin columnas" in log or "vacío" in log
    assert "no tiene valor guardado" in log
    assert pngs(out) == EXPECTED_VAL


def test_solo_subconjuntos_genera_con_aviso(runs_dir, tmp_path, cwd, capsys):
    make_run(runs_dir / "_smoke", "b2_malexnet", subset=0.02, sep=1.0, seed=1)
    make_run(runs_dir / "_smoke", "k3_mnv3_ft", subset=0.02, sep=2.0, seed=2)
    out = tmp_path / "figs"
    assert MF.main(["--runs-dir", str(runs_dir), "--out", str(out), "--include-smoke"]) == 0
    assert pngs(out) == EXPECTED_VAL
    assert "no son concluyentes" in capsys.readouterr().out
    # sin --include-smoke no hay runs completos: no se genera nada
    out2 = tmp_path / "figs2"
    assert MF.main(["--runs-dir", str(runs_dir), "--out", str(out2)]) == 1


# ---------------------------------------------------------------------------
# Unitarias
# ---------------------------------------------------------------------------

def test_three_state_counts_a_mano():
    df = pd.DataFrame({"y": [0, 0, 0, 1, 1, 1, 1],
                       "p": [0.1, 0.4, 0.9, 0.2, 0.5, 0.69, 0.7]})
    counts = MF.three_state_counts(df, 0.3, 0.7)
    # libres: 0.1 → libre, 0.4 → verificar, 0.9 → ocupado
    # ocupados: 0.2 → libre, 0.5 y 0.69 → verificar, 0.7 → ocupado (p ≥ t_H)
    assert counts.tolist() == [[1, 1, 1], [1, 2, 1]]


def test_split_predictions_val_aplica_la_temperatura(runs_dir):
    T = 2.0
    thr = {"comercial": {"t_L": 0.4, "t_H": 0.6}, "uvg": {"t_L": 0.4, "t_H": 0.6}}
    run = make_run(runs_dir, "k6_mnv3_ft_seeds", calibration={"temperature": T, "thresholds": thr})
    rec = RT.load_records(runs_dir)[0]
    df = MF.split_predictions(rec, "val")
    np.testing.assert_allclose(df["p"].to_numpy(), sigmoid(run["logits"][:, 1] / T), rtol=1e-6)
    assert set(df.columns) == {"y", "p", "camera", "weather"}
    assert MF.available_splits([rec]) == ["val"]


def test_value_vs_t_L_cubre_hasta_t_H(runs_dir):
    run = make_run(runs_dir, "k3_mnv3_ft")
    cfg = RT.load_scenarios()["comercial"]
    curve = MF.value_vs_t_L(run["y"], run["p"], cfg, 0.6)
    assert curve["t_L"].iloc[0] == 0.0 and curve["t_L"].iloc[-1] == pytest.approx(0.6)
    assert len(curve) == 61


# ---------------------------------------------------------------------------
# Invariante: test no elige el mejor run
# ---------------------------------------------------------------------------

def test_metrics_test_alto_no_cambia_el_mejor_run(runs_dir, tmp_path, cwd, capsys):
    good = make_run(runs_dir, "k3_mnv3_ft", sep=2.5, noise=1.0, seed=1)
    bad = make_run(runs_dir, "k2_mnv3_head", sep=0.8, noise=1.5, seed=2, extra_split_preds=("test",))
    assert good["values"]["comercial"] > bad["values"]["comercial"]
    (bad["run_dir"] / "metrics_test.json").write_text(
        __import__("json").dumps(metrics_split(0.9999, 0.0, 1e9)), encoding="utf-8")
    # predicciones de test perfectas en el run peor en val
    ys, lg, cam, wea, dt, fl = synthetic_preds(200, 9.0, 0.1, 5)
    np.savez_compressed(bad["run_dir"] / "test_preds.npz", y=ys, p=sigmoid(lg[:, 1]), logits=lg,
                        camera=cam, weather=wea, date=dt, path=fl)
    out = tmp_path / "figs"
    assert MF.main(["--runs-dir", str(runs_dir), "--out", str(out)]) == 0
    log = capsys.readouterr().out
    m = re.search(r"Mejor run \(valor comercial en val\): (\S+)", log)
    assert m and m.group(1) == good["run_id"]
    assert split_figures("test") <= pngs(out)


def test_make_figures_no_lee_datos_ni_particiones():
    src = (REPO / "scripts" / "make_figures.py").read_text(encoding="utf-8")
    assert "cupo.data" not in src and "load_split" not in src and "allow_test" not in src
    # la selección del mejor run se reutiliza de results_table (solo val)
    assert MF.best_record is RT.best_record and MF.representatives is RT.representatives


# ---------------------------------------------------------------------------
# Runs reales (copias de runs/_smoke y runs/_obsoleto)
# ---------------------------------------------------------------------------

REAL_RUNS = paths.runs_dir()


@pytest.mark.skipif(not (REAL_RUNS / "_smoke").is_dir(), reason="no hay runs/_smoke")
def test_runs_reales_copiados(tmp_path, cwd):
    root = tmp_path / "runs"
    before = snapshot(REAL_RUNS)
    shutil.copytree(REAL_RUNS / "_smoke", root / "_smoke")
    obsolete = REAL_RUNS / "_obsoleto"
    if obsolete.is_dir():
        for d in obsolete.iterdir():
            if d.is_dir():
                shutil.copytree(d, root / d.name)
    out = tmp_path / "figs"
    rc = MF.main(["--runs-dir", str(root), "--out", str(out), "--include-smoke", "--with-table",
                  "--table-out", str(tmp_path / "tabla")])
    assert rc == 0
    assert snapshot(REAL_RUNS) == before
    made = pngs(out)
    assert {"valor_vs_umbral.png", "tornado_comercial.png", "matrices_confusion_val.png"} <= made
    for name in made:
        assert_valid_png(out / name)
    assert (tmp_path / "tabla" / "iteraciones.md").is_file()


# ---------------------------------------------------------------------------
# Un solo comando (como se corre desde la terminal)
# ---------------------------------------------------------------------------

def test_un_comando_deja_tabla_y_figuras(runs_dir, tmp_path, cwd):
    import subprocess
    import sys

    three_runs(runs_dir)
    out, table_out = tmp_path / "figs", tmp_path / "tabla"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_figures.py"), "--runs-dir", str(runs_dir),
         "--out", str(out), "--with-table", "--table-out", str(table_out)],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert pngs(out) == EXPECTED_VAL
    assert (table_out / "iteraciones.md").is_file() and (table_out / "iteraciones.csv").is_file()
    assert list(cwd.iterdir()) == []

    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "results_table.py"), "--runs-dir", str(runs_dir),
         "--out", str(tmp_path / "tabla2")],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert (tmp_path / "tabla2" / "iteraciones_resumen.md").is_file()
    assert (tmp_path / "tabla2" / "iteraciones.csv").read_bytes() == (table_out / "iteraciones.csv").read_bytes()
