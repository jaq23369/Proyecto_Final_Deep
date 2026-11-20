"""Pruebas de scripts/uncertainty.py (IC95 por bootstrap en test y OOD).

Casi todo usa runs sintéticos en una carpeta temporal: las predicciones de test y OOD se
escriben con ``cupo.evaluate.write_split_outputs`` (igual que en el congelamiento), así que
``metrics_<split>.json`` y ``value_<split>.json`` sirven de referencia independiente para la
estimación puntual y los IC95. Nada se entrena ni se lee del dataset.

La última prueba usa los runs reales (``runs/*/test_preds.npz``) si existen: regenera el
reporte con pocas réplicas en una carpeta temporal y compara las estimaciones puntuales con
``results/iteraciones.csv`` y ``results/incertidumbre.csv`` (solo lectura).
"""

from __future__ import annotations

import ast
import importlib.util
import math
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import f1_score

from cupo import evaluate as E
from cupo import metrics as M
from cupo import paths
from cupo.runs import load_json, load_npz, save_json, save_npz
from cupo.value_model import compute_value

from results_helpers import manual_rates, snapshot

REPO = paths.repo_root()
SCRIPT = REPO / "scripts" / "uncertainty.py"


def _load_module():
    name = "cupo_scripts_uncertainty"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


U = _load_module()
SCEN = E.load_scenarios("comercial")
THR = {"comercial": {"t_L": 0.3, "t_H": 0.7}, "uvg": {"t_L": 0.2, "t_H": 0.8}}
THR_CAL = {"comercial": {"t_L": 0.25, "t_H": 0.75}, "uvg": {"t_L": 0.15, "t_H": 0.85}}
N_IMAGES = {"test": 300, "ood": 200}
N_BOOT = 40
FINAL, B1, B2, SEEDS = "mnv3_ft_cost", "b1_hog_lr", "b2_malexnet", "mnv3_ft_seeds"


# ---------------------------------------------------------------------------
# Runs sintéticos
# ---------------------------------------------------------------------------

def images(split: str, busy_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Etiquetas y rutas de una partición; las mismas para todos los runs."""
    rng = np.random.default_rng(0 if split == "test" else 100)
    n = N_IMAGES[split]
    y = rng.binomial(1, busy_frac, n).astype(int)
    files = np.array([f"{split}/2015-11-20/camera{i % 3 + 1}/slot_{i:04d}.jpg" for i in range(n)])
    return y, files


def noisy_z(y: np.ndarray, sep: float, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(y == 1, sep, -sep) + rng.normal(0.0, noise, y.size)


def split_df(y: np.ndarray, files: np.ndarray) -> pd.DataFrame:
    n = y.size
    return pd.DataFrame({
        "label": y, "path": files,
        "camera": [str(i % 3 + 1) for i in range(n)],
        "weather": ["SUNNY" if i % 2 else "OVERCAST" for i in range(n)],
        "date": ["2015-11-20"] * n, "time": ["12:00"] * n,
        "slot_id": [str(i) for i in range(n)],
    })


def make_run(root: Path, name: str, seed: int = 42, *, sep: float = 2.0, noise: float = 1.5,
             z: dict[str, np.ndarray] | None = None, n_boot: int | None = N_BOOT,
             thresholds: dict | None = None, calibration: dict | None = None,
             splits: tuple[str, ...] = ("test", "ood"), subset: float | None = None,
             perm: dict[str, np.ndarray] | None = None, run_id: str | None = None) -> Path:
    """``root/<run_id>/`` con config.yaml, thresholds.json (y calibration.json) y las
    predicciones de cada partición escritas por ``cupo.evaluate.write_split_outputs``."""
    run_dir = Path(root) / (run_id or f"{name}_s{seed}_20261112-1200")
    run_dir.mkdir(parents=True)
    cfg: dict = {"name": name, "seed": seed}
    if subset is not None:
        cfg["data"] = {"subset": subset}
    if n_boot is not None:
        cfg["eval"] = {"n_boot": n_boot}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    thresholds = thresholds or THR
    save_json(run_dir / "thresholds.json", thresholds)
    temperature, op_thr = 1.0, thresholds
    if calibration is not None:
        save_json(run_dir / "calibration.json", calibration)
        temperature, op_thr = float(calibration["temperature"]), calibration["thresholds"]
    for k, split in enumerate(splits):
        y, files = images(split)
        zs = z[split] if z is not None else noisy_z(y, sep, noise, seed * 10 + k)
        logits = np.stack([np.zeros_like(zs), zs], axis=1).astype(np.float32)
        df = split_df(y, files)
        if perm is not None and split in perm:
            df, logits = df.iloc[perm[split]].reset_index(drop=True), logits[perm[split]]
        p = E.probs_from_logits(logits, temperature)
        scen = {s: c for s, c in SCEN.items() if s in op_thr}
        E.write_split_outputs(run_dir, split, df, logits, p, op_thr, scen,
                              n_boot=n_boot or E.DEFAULT_N_BOOT, seed=0, tag=split)
    return run_dir


def perfect_z() -> dict[str, np.ndarray]:
    return {sp: np.where(images(sp)[0] == 1, 8.0, -8.0) for sp in ("test", "ood")}


def flawed_z(frac_busy_free: float = 0.25) -> dict[str, np.ndarray]:
    """Igual al perfecto salvo una fracción de ocupados mostrados libres."""
    out = {}
    for sp in ("test", "ood"):
        y = images(sp)[0]
        zz = np.where(y == 1, 8.0, -8.0)
        busy = np.flatnonzero(y == 1)
        zz[busy[: int(len(busy) * frac_busy_free)]] = -8.0
        out[sp] = zz
    return out


def standard_runs(root: Path, *, final_z=None, b1_z=None, b2_z=None, seeds: bool = False) -> Path:
    runs = root / "runs"
    make_run(runs, B1, 42, sep=1.2, noise=1.5, z=b1_z)
    make_run(runs, B2, 42, sep=2.0, noise=1.5, z=b2_z)
    make_run(runs, FINAL, 42, sep=3.0, noise=1.5, z=final_z,
             calibration={"temperature": 1.6, "thresholds": THR_CAL})
    if seeds:
        for s in (42, 43, 44):
            make_run(runs, SEEDS, s, sep=2.5, noise=1.5 + 0.3 * (s - 42))
    return runs


def load_groups(runs: Path, n_boot: int | None = None):
    return U.load_groups(runs, n_boot)


def build(runs: Path, iterations: Path, seed: int = 0, refs=(B1, B2), n_boot=None) -> pd.DataFrame:
    return U.build_table(load_groups(runs, n_boot), FINAL, list(refs), SCEN, seed, iterations)


def cli(runs: Path, out: Path, iterations: Path, *extra: str) -> int:
    return U.main(["--allow-test", "--runs-dir", str(runs), "--out", str(out),
                   "--iterations", str(iterations), "--final", FINAL, "--refs", B1, B2, *extra])


def row(table: pd.DataFrame, kind: str, group: str, split: str, metric: str,
        reference: str = "") -> pd.Series:
    sel = table[(table["kind"] == kind) & (table["group"] == group) & (table["split"] == split)
                & (table["metric"] == metric) & (table["reference"].fillna("") == reference)]
    assert len(sel) == 1, f"{kind} {group} {split} {metric} {reference}: {len(sel)} filas"
    return sel.iloc[0]


def evaluate_reference(run_dir: Path, split: str) -> dict[str, tuple[float, float, float]]:
    """{métrica: (estimación, IC lo, IC hi)} según los JSON que escribió cupo.evaluate."""
    m = load_json(run_dir / f"metrics_{split}.json")
    out = {k: (m["binary"][k], *m["ci95"][k]) for k in ("f1_macro", "rate_busy_as_free")}
    for scen, val in m["value"].items():
        out[f"value_{scen}"] = (val, *m["value_ci95"][scen])
    return out


def iterations_csv_from_metrics(runs: Path, path: Path, groups: dict[str, list[Path]],
                                tweak: dict[str, float] | None = None) -> Path:
    """iteraciones.csv sintético con las columnas test_/ood_ que lee el script (BOM incluido)."""
    rows = []
    for group, run_dirs in groups.items():
        r = {"iteration": U.iteration_id(group), "group": group}
        for split in ("test", "ood"):
            per = [evaluate_reference(d, split) for d in run_dirs]
            for metric in ("f1_macro", "rate_busy_as_free", "value_comercial"):
                vals = [e[metric][0] for e in per]
                r[f"{split}_{metric}"] = statistics.fmean(vals)
                r[f"{split}_{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else None
        rows.append(r)
    df = pd.DataFrame(rows)
    for col, delta in (tweak or {}).items():
        group, c = col.split(":")
        df.loc[df["group"] == group, c] += delta
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_dirs_by_group(runs: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for d in sorted(runs.iterdir()):
        name = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))["name"]
        out.setdefault(name, []).append(d)
    return out


# ---------------------------------------------------------------------------
# Guarda de test y salidas
# ---------------------------------------------------------------------------

def test_parse_args_requires_allow_test(capsys):
    with pytest.raises(SystemExit) as exc:
        U.parse_args(["--runs-dir", "runs", "--out", "results"])
    assert exc.value.code == 2
    assert "--allow-test" in capsys.readouterr().err
    assert U.parse_args(["--allow-test"]).allow_test is True


def test_main_without_allow_test_writes_nothing(tmp_path):
    runs = standard_runs(tmp_path)
    out = tmp_path / "out"
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit) as exc:
        U.main(["--runs-dir", str(runs), "--out", str(out)])
    assert exc.value.code == 2
    assert not out.exists()
    assert snapshot(tmp_path) == before


def test_cli_subprocess_without_allow_test_exits_2(tmp_path):
    out = tmp_path / "out"
    res = subprocess.run([sys.executable, str(SCRIPT), "--runs-dir", str(tmp_path), "--out", str(out)],
                         cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=300)
    assert res.returncode == 2, res.stderr[-1500:]
    assert "--allow-test" in res.stderr
    assert not out.exists()


def test_main_writes_only_to_out(tmp_path):
    runs = standard_runs(tmp_path, seeds=True)
    iterations = iterations_csv_from_metrics(runs, tmp_path / "iteraciones.csv",
                                             run_dirs_by_group(runs))
    out = tmp_path / "salida"
    protected = {name: snapshot(REPO / name) for name in ("models", "runs", "results")}
    runs_before, it_before = snapshot(runs), iterations.read_bytes()

    assert cli(runs, out, iterations) == 0

    assert sorted(p.name for p in out.iterdir()) == ["incertidumbre.csv", "incertidumbre.md"]
    assert snapshot(runs) == runs_before
    assert iterations.read_bytes() == it_before
    for name, snap in protected.items():
        assert snapshot(REPO / name) == snap, f"se modificó {name}/ del repo"
    table = pd.read_csv(out / "incertidumbre.csv")
    assert list(table.columns) == list(U.CSV_COLUMNS)


def test_script_never_loads_dataset_or_checkpoints():
    """Solo lee *_preds.npz, config y punto de operación: ni load_split ni best.pt/joblib."""
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    modules, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.add(node.module)
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
    assert not modules & {"cupo.data", "cupo.calibrate", "cupo.thresholds", "cupo.train", "torch"}
    assert not names & {"load_split", "load_run", "optimize", "choose_thresholds", "data"}
    for forbidden in ("best.pt", "model.joblib", "val_preds", "load_split("):
        assert forbidden not in src, forbidden
    assert U.SPLITS == ("test", "ood")


# ---------------------------------------------------------------------------
# Temperatura y punto de operación
# ---------------------------------------------------------------------------

def test_uses_calibration_temperature_and_thresholds(tmp_path):
    run_dir = make_run(tmp_path, FINAL, calibration={"temperature": 2.5, "thresholds": THR_CAL})
    run = U.load_run_preds(run_dir, None)
    assert run.temperature == 2.5
    assert run.thresholds == THR_CAL
    for split in ("test", "ood"):
        logits = load_npz(run_dir / f"{split}_preds.npz")["logits"]
        np.testing.assert_allclose(run.splits[split].p, E.probs_from_logits(logits, 2.5), rtol=0, atol=0)
        assert not np.allclose(run.splits[split].p, E.probs_from_logits(logits, 1.0))


def test_uses_thresholds_json_without_calibration(tmp_path):
    run_dir = make_run(tmp_path, B2)
    run = U.load_run_preds(run_dir, None)
    assert run.temperature == 1.0
    assert run.thresholds == THR
    logits = load_npz(run_dir / "test_preds.npz")["logits"]
    np.testing.assert_array_equal(run.splits["test"].p, E.probs_from_logits(logits, 1.0))


def test_p_is_recomputed_from_logits_not_stored_p(tmp_path):
    run_dir = make_run(tmp_path, FINAL, calibration={"temperature": 2.0, "thresholds": THR_CAL})
    data = load_npz(run_dir / "test_preds.npz")
    data["p"] = np.full_like(data["p"], 0.5)
    save_npz(run_dir / "test_preds.npz", **data)
    run = U.load_run_preds(run_dir, None)
    np.testing.assert_array_equal(run.splits["test"].p, E.probs_from_logits(data["logits"], 2.0))


# ---------------------------------------------------------------------------
# Estimación puntual e IC95
# ---------------------------------------------------------------------------

def test_hand_computed_small_case():
    """y = [1,1,1,0,0], p = [0.9,0.6,0.2,0.1,0.55], umbrales (0.3, 0.7).

    A 0.5: TP=2, FN=1, FP=1, TN=1 → F1 ocupado 4/6, F1 libre 2/4, macro 7/12;
    ocupado→libre 1/3. Tres estados: q_OL = q_OV = q_OO = 1/3, q_LL = q_LV = 1/2, q_LO = 0.
    """
    y = np.array([1, 1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.1, 0.55])
    sp = U.SplitPreds(y=y, p=p, path=np.array([f"x{i}" for i in range(5)]))
    run = U.RunPreds(run_dir=Path("r"), name="x", temperature=1.0,
                     thresholds={"comercial": {"t_L": 0.3, "t_H": 0.7}}, n_boot=20,
                     splits={"test": sp, "ood": sp})
    est = U.run_estimates(run, "test", SCEN, seed=0)
    assert est["f1_macro"][0] == pytest.approx(7 / 12)
    assert est["rate_busy_as_free"][0] == pytest.approx(1 / 3)
    rates = {"q_OL": 1 / 3, "q_OV": 1 / 3, "q_OO": 1 / 3, "q_LL": 0.5, "q_LV": 0.5, "q_LO": 0.0}
    assert est["value_comercial"][0] == pytest.approx(compute_value(rates, SCEN["comercial"])["objective"])
    assert "value_uvg" not in est  # el run no tiene umbrales de uvg


@pytest.mark.parametrize("calibrated", [False, True])
def test_estimates_and_cis_match_cupo_evaluate(tmp_path, calibrated):
    calib = {"temperature": 1.8, "thresholds": THR_CAL} if calibrated else None
    run_dir = make_run(tmp_path, FINAL, calibration=calib)
    run = U.load_run_preds(run_dir, None)
    thr = THR_CAL if calibrated else THR
    for split in ("test", "ood"):
        est = U.run_estimates(run, split, SCEN, seed=0)
        ref = evaluate_reference(run_dir, split)
        assert set(est) == set(ref) == set(U.METRICS)
        for metric in U.METRICS:
            assert est[metric] == pytest.approx(ref[metric], rel=1e-12, abs=1e-12), (split, metric)
        # Cálculo independiente de la estimación puntual
        sp = run.splits[split]
        assert est["f1_macro"][0] == pytest.approx(f1_score(sp.y, sp.p >= 0.5, average="macro"))
        assert est["rate_busy_as_free"][0] == pytest.approx(np.mean(sp.p[sp.y == 1] < 0.5))
        for scen in ("comercial", "uvg"):
            t = thr[scen]
            manual = compute_value(manual_rates(sp.y, sp.p, t["t_L"], t["t_H"]), SCEN[scen])
            assert est[f"value_{scen}"][0] == pytest.approx(manual["objective"], rel=1e-12)
        # Los IC95 son los de metric_cis / value_report con la misma semilla
        cis = E.metric_cis(sp.y, sp.p, N_BOOT, seed=0)
        assert list(est["f1_macro"][1:]) == cis["f1_macro"]
        assert list(est["rate_busy_as_free"][1:]) == cis["rate_busy_as_free"]


def test_cis_deterministic_contain_estimate_and_depend_on_seed(tmp_path):
    runs = standard_runs(tmp_path, seeds=True)
    it = tmp_path / "no_existe.csv"
    t1, t2 = build(runs, it, seed=0), build(runs, it, seed=0)
    pd.testing.assert_frame_equal(t1, t2)
    metric_rows = t1[t1["kind"] == "metric"]
    single = metric_rows[metric_rows["n_runs"] == 1]
    assert ((single["ci_lo"] <= single["estimate"] + 1e-9)
            & (single["estimate"] <= single["ci_hi"] + 1e-9)).all()
    assert (metric_rows["ci_lo"] <= metric_rows["ci_hi"]).all()
    assert (t1["seed"] == 0).all()
    t3 = build(runs, it, seed=7)
    pd.testing.assert_series_equal(t1["estimate"], t3["estimate"])
    assert not np.allclose(t1["ci_lo"], t3["ci_lo"])
    assert (t3["seed"] == 7).all()


def test_n_boot_from_config_and_override(tmp_path):
    runs = tmp_path / "runs"
    run_dir = make_run(runs, FINAL, n_boot=23)
    run = U.load_run_preds(run_dir, None)
    assert run.n_boot == 23
    table = U.build_table({FINAL: [run]}, FINAL, [], SCEN, 0, tmp_path / "x.csv")
    assert (table["n_boot"] == 23).all()
    sp = run.splits["test"]
    r = row(table, "metric", FINAL, "test", "f1_macro")
    assert [r["ci_lo"], r["ci_hi"]] == E.metric_cis(sp.y, sp.p, 23, 0)["f1_macro"]
    assert U.load_run_preds(run_dir, 17).n_boot == 17
    # Sin eval.n_boot en la config: el valor por defecto de cupo.evaluate
    other = make_run(tmp_path / "otros", B2, n_boot=None, splits=())
    for split in ("test", "ood"):
        (other / f"{split}_preds.npz").write_bytes((run_dir / f"{split}_preds.npz").read_bytes())
    assert U.load_run_preds(other, None).n_boot == E.DEFAULT_N_BOOT


# ---------------------------------------------------------------------------
# Diferencia pareada
# ---------------------------------------------------------------------------

def test_paired_delta_identical_models_is_zero(tmp_path):
    y, _ = images("test")
    p = E.probs_from_logits(np.stack([np.zeros(y.size), noisy_z(y, 2, 1.5, 0)], 1))
    fn = U.objective_fn(SCEN["comercial"], 0.3, 0.7)
    res = U.paired_delta(y, p, p.copy(), fn, fn, n_boot=50, seed=0)
    assert res == {"estimate": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_better": 0.0}

    z = {sp: noisy_z(images(sp)[0], 2.0, 1.5, 3) for sp in ("test", "ood")}
    runs = tmp_path / "runs"
    make_run(runs, FINAL, z=z)
    make_run(runs, B1, z=z)
    table = build(runs, tmp_path / "x.csv", refs=[B1])
    deltas = table[table["kind"] == "delta"]
    assert len(deltas) == 4  # 2 particiones × 2 escenarios
    assert (deltas["estimate"] == 0).all()
    assert (deltas["ci_lo"] == 0).all() and (deltas["ci_hi"] == 0).all()
    assert (deltas["p_better"] == 0).all()


def test_paired_delta_matches_manual_resampling():
    y, _ = images("test")
    pa = E.probs_from_logits(np.stack([np.zeros(y.size), noisy_z(y, 3, 1.5, 1)], 1))
    pb = E.probs_from_logits(np.stack([np.zeros(y.size), noisy_z(y, 1.5, 1.5, 2)], 1))
    fa = U.objective_fn(SCEN["comercial"], 0.25, 0.75)
    fb = U.objective_fn(SCEN["comercial"], 0.3, 0.7)
    res = U.paired_delta(y, pa, pb, fa, fb, n_boot=60, seed=5)
    rng = np.random.default_rng(5)
    deltas, a_only = [], []
    for _ in range(60):
        idx = rng.integers(0, y.size, size=y.size)
        deltas.append(fa(y[idx], pa[idx]) - fb(y[idx], pb[idx]))
        a_only.append(fa(y[idx], pa[idx]))
    assert res["estimate"] == pytest.approx(fa(y, pa) - fb(y, pb))
    assert [res["ci_lo"], res["ci_hi"]] == pytest.approx(list(np.percentile(deltas, [2.5, 97.5])))
    assert res["p_better"] == pytest.approx(np.mean(np.array(deltas) > 0))
    # Mismas remuestras que el IC95 marginal de cupo.metrics.bootstrap_ci
    assert list(np.percentile(a_only, [2.5, 97.5])) == pytest.approx(
        list(M.bootstrap_ci(fa, y, pa, n_boot=60, seed=5)))


def test_paired_delta_dominating_model(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs, FINAL, z=perfect_z())
    make_run(runs, B1, z=flawed_z(0.25))
    table = build(runs, tmp_path / "x.csv", refs=[B1])
    for split in ("test", "ood"):
        for scen in ("comercial", "uvg"):
            d = row(table, "delta", FINAL, split, f"value_{scen}", B1)
            a = row(table, "metric", FINAL, split, f"value_{scen}")
            b = row(table, "metric", B1, split, f"value_{scen}")
            assert d["estimate"] == pytest.approx(a["estimate"] - b["estimate"])
            assert d["estimate"] > 0
            assert d["p_better"] == 1.0
            assert d["ci_lo"] > 0
            assert d["n_boot"] == N_BOOT and d["n_runs"] == 1


def test_pairing_by_path_is_order_invariant(tmp_path):
    z_ref = {sp: noisy_z(images(sp)[0], 1.5, 1.5, 9) for sp in ("test", "ood")}
    z_fin = {sp: noisy_z(images(sp)[0], 3.0, 1.5, 8) for sp in ("test", "ood")}
    rng = np.random.default_rng(123)
    perm = {sp: rng.permutation(N_IMAGES[sp]) for sp in ("test", "ood")}
    a, b = tmp_path / "a", tmp_path / "b"
    make_run(a, FINAL, z=z_fin)
    make_run(a, B1, z=z_ref)
    make_run(b, FINAL, z=z_fin)
    make_run(b, B1, z=z_ref, perm=perm)
    assert not np.array_equal(load_npz(next(a.glob("b1*")) / "test_preds.npz")["path"],
                              load_npz(next(b.glob("b1*")) / "test_preds.npz")["path"])
    ta, tb = build(a, tmp_path / "x.csv", refs=[B1]), build(b, tmp_path / "x.csv", refs=[B1])
    cols = ["split", "metric", "estimate", "ci_lo", "ci_hi", "p_better"]
    pd.testing.assert_frame_equal(ta[ta["kind"] == "delta"][cols].reset_index(drop=True),
                                  tb[tb["kind"] == "delta"][cols].reset_index(drop=True))
    # La estimación puntual de cada modelo tampoco depende del orden
    pd.testing.assert_series_equal(ta["estimate"], tb["estimate"])


def test_different_image_sets_raise_clear_error(tmp_path):
    with pytest.raises(ValueError, match="mismo conjunto de imágenes"):
        U.alignment(np.array(["a", "b", "c"]), np.array(["a", "b", "d"]))
    with pytest.raises(ValueError, match="repetidas"):
        U.alignment(np.array(["a", "b", "b"]), np.array(["a", "b", "b"]))
    np.testing.assert_array_equal(U.alignment(np.array(["a", "b", "c"]), np.array(["c", "a", "b"])),
                                  [1, 2, 0])
    runs = standard_runs(tmp_path)
    bad = next(runs.glob("b2*")) / "ood_preds.npz"
    data = load_npz(bad)
    data["path"] = np.array([f"otra/{p}" for p in data["path"]])
    save_npz(bad, **data)
    with pytest.raises(ValueError, match="mismo conjunto de imágenes"):
        U.main(["--allow-test", "--runs-dir", str(runs), "--out", str(tmp_path / "out"),
                "--iterations", str(tmp_path / "x.csv"), "--final", FINAL])
    assert not (tmp_path / "out").exists()


def test_different_labels_for_same_images_raise(tmp_path):
    runs = standard_runs(tmp_path)
    bad = next(runs.glob("b2*")) / "test_preds.npz"
    data = load_npz(bad)
    data["y"] = 1 - data["y"]
    save_npz(bad, **data)
    with pytest.raises(ValueError, match="etiquetas distintas"):
        U.check_same_images(load_groups(runs))


# ---------------------------------------------------------------------------
# Varias semillas
# ---------------------------------------------------------------------------

def test_seed_group_mean_std_and_mean_ci(tmp_path):
    runs = standard_runs(tmp_path, seeds=True)
    table = build(runs, tmp_path / "x.csv")
    seed_dirs = sorted(runs.glob(f"{SEEDS}_*"))
    assert len(seed_dirs) == 3
    for split in ("test", "ood"):
        per = [evaluate_reference(d, split) for d in seed_dirs]
        for metric in U.METRICS:
            r = row(table, "metric", SEEDS, split, metric)
            est, lo, hi = ([e[metric][k] for e in per] for k in range(3))
            assert r["estimate"] == pytest.approx(statistics.fmean(est), rel=1e-12)
            assert r["estimate_std"] == pytest.approx(statistics.stdev(est), rel=1e-9)
            assert r["ci_lo"] == pytest.approx(statistics.fmean(lo), rel=1e-12)
            assert r["ci_hi"] == pytest.approx(statistics.fmean(hi), rel=1e-12)
            assert r["n_runs"] == 3
            assert r["run_ids"] == "; ".join(d.name for d in seed_dirs)
        single = row(table, "metric", B1, split, "f1_macro")
        assert single["n_runs"] == 1 and math.isnan(single["estimate_std"])


def test_seed_group_cannot_be_paired(tmp_path):
    runs = standard_runs(tmp_path, seeds=True)
    with pytest.raises(ValueError, match="necesita uno"):
        U.build_table(load_groups(runs), SEEDS, [B1], SCEN, 0, tmp_path / "x.csv")


# ---------------------------------------------------------------------------
# Verificación contra iteraciones.csv
# ---------------------------------------------------------------------------

def test_verification_column(tmp_path, capsys):
    runs = standard_runs(tmp_path, seeds=True)
    groups = run_dirs_by_group(runs)
    groups.pop(B2)  # B2 ausente de iteraciones.csv
    it = iterations_csv_from_metrics(runs, tmp_path / "iteraciones.csv", groups,
                                     tweak={f"{B1}:test_f1_macro": 0.01,
                                            f"{SEEDS}:ood_value_comercial_std": 5.0})
    table = build(runs, it)
    capsys.readouterr()
    metric = table[table["kind"] == "metric"]

    def match(group, split, m):
        return row(table, "metric", group, split, m)["matches_iteraciones"]

    assert match(B1, "test", "f1_macro") == False  # noqa: E712
    assert bool(match(B1, "ood", "f1_macro")) is True
    assert match(SEEDS, "ood", "value_comercial") == False  # noqa: E712 (desviación distinta)
    assert bool(match(SEEDS, "test", "value_comercial")) is True
    assert bool(match(FINAL, "test", "value_comercial")) is True
    assert pd.isna(match(B2, "test", "f1_macro"))  # ausente
    assert metric[metric["metric"] == "value_uvg"]["matches_iteraciones"].isna().all()
    assert table[table["kind"] == "delta"]["matches_iteraciones"].isna().all()
    checked = metric["matches_iteraciones"].dropna()
    # 3 grupos verificables × 2 particiones × 3 métricas
    assert len(checked) == 18 and int(checked.astype(bool).sum()) == 16
    r = row(table, "metric", B1, "test", "f1_macro")
    assert r["iteraciones_value"] == pytest.approx(r["estimate"] + 0.01)

    lines = U.verification_lines(table, it)
    assert lines[0].startswith("- 16 de 18 estimaciones puntuales")
    assert any(ln.startswith(f"- No coincide: {B1} test_f1_macro") for ln in lines)
    assert any(ln.startswith(f"- No coincide: {SEEDS} ood_value_comercial") for ln in lines)


def test_verification_with_missing_iterations_csv(tmp_path, capsys):
    runs = standard_runs(tmp_path)
    missing = tmp_path / "no_existe.csv"
    table = build(runs, missing)
    assert "no existe" in capsys.readouterr().out
    assert table["matches_iteraciones"].isna().all()
    assert table["iteraciones_value"].isna().all()
    assert U.verification_lines(table, missing)[0].startswith("- No se pudo verificar")


# ---------------------------------------------------------------------------
# Markdown y CSV
# ---------------------------------------------------------------------------

def test_markdown_tables_and_conclusions(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs, FINAL, z=perfect_z(), calibration={"temperature": 1.5, "thresholds": THR})
    make_run(runs, B1, z=flawed_z(0.30))
    make_run(runs, B2, z=flawed_z(0.10))
    make_run(runs, "efficientnet_b0", z=perfect_z())  # no supera al final: no se menciona
    for s in (42, 43, 44):
        make_run(runs, SEEDS, s, sep=2.5, noise=1.5 + 0.4 * (s - 42))
    it = iterations_csv_from_metrics(runs, tmp_path / "iteraciones.csv", run_dirs_by_group(runs))
    out = tmp_path / "out"
    assert cli(runs, out, it) == 0
    md = (out / "incertidumbre.md").read_text(encoding="utf-8")
    csv = pd.read_csv(out / "incertidumbre.csv")

    for heading in ("# Incertidumbre en test y OOD", "## Test: CNR-EXT, fechas de test (n = 300)",
                    "## OOD: CNRPark A/B (n = 200)",
                    "## Diferencia pareada: modelo final frente a los baselines",
                    "## Verificación", "## Conclusiones", "## Notas"):
        assert heading in md, heading
    assert f"IC95 por bootstrap ({N_BOOT} réplicas, semilla 0)" in md
    assert f"**{FINAL}** — modelo final" in md
    assert f"**B1** ({B1})" in md and f"**B2** ({B2})" in md
    for ref in ("B1", "B2"):
        for part in ("Test", "OOD"):
            assert f"| {FINAL} − {ref} | {part} |" in md
    # Celda de F1 del final en test: '0.xxxx [lo; hi]'
    r = csv[(csv["kind"] == "metric") & (csv["group"] == FINAL) & (csv["split"] == "test")
            & (csv["metric"] == "f1_macro")].iloc[0]
    assert f"{r['estimate']:.4f} [{r['ci_lo']:.4f}; {r['ci_hi']:.4f}]" in md
    # Grupo de semillas con media ± desviación
    seeds_line = next(ln for ln in md.splitlines() if ln.startswith(f"| **{SEEDS}**"))
    assert seeds_line.count("±") == 4
    assert "- 30 de 30 estimaciones puntuales" in md

    conclusions = md.split("## Conclusiones")[1].split("## Notas")[0]
    for where in ("En OOD (CNRPark A/B)", "En test"):
        for ref in ("B1", "B2"):
            sentence = next(ln for ln in conclusions.splitlines()
                            if ln.startswith(f"- {where}, los IC95") and f"y de {ref} [" in ln)
            assert "no se solapan" in sentence and "que excluye el 0" in sentence
            assert f"{FINAL} supera a {ref} en 100.0 % de las remuestras" in sentence
    assert f"- {SEEDS}: en test la desviación entre semillas" in conclusions
    assert "superan en valor" not in conclusions and "supera en valor" not in conclusions


def test_markdown_reports_groups_above_final_and_overlap(tmp_path):
    runs = tmp_path / "runs"
    z_final = flawed_z(0.05)
    make_run(runs, FINAL, z=z_final)
    make_run(runs, B1, z=z_final)  # igual al final: IC solapados y Δ = 0
    make_run(runs, "efficientnet_b0", z=perfect_z())
    table = build(runs, tmp_path / "x.csv", refs=[B1])
    lines = U.conclusions(table, FINAL, [B1])
    ood_b1 = next(ln for ln in lines if ln.startswith("- En OOD (CNRPark A/B)"))
    assert "] se solapan." in ood_b1 and "que incluye el 0" in ood_b1
    assert "en 0.0 % de las remuestras" in ood_b1
    above = next(ln for ln in lines if ln.startswith("- En OOD, "))
    assert "efficientnet_b0 (Q" in above and "supera en valor comercial puntual" in above


# ---------------------------------------------------------------------------
# Runs incompletos o fuera de la tabla
# ---------------------------------------------------------------------------

def test_run_without_split_npz_is_skipped(tmp_path):
    runs = standard_runs(tmp_path)
    make_run(runs, "malexnet_aug", splits=("test",))
    assert not U.is_candidate(next(runs.glob("malexnet_aug*")))
    out = tmp_path / "out"
    assert cli(runs, out, tmp_path / "x.csv") == 0
    csv = pd.read_csv(out / "incertidumbre.csv")
    assert "malexnet_aug" not in set(csv["group"])
    assert {B1, B2, FINAL} <= set(csv["group"])


def test_final_without_ood_npz_fails_with_clear_message(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs, B1)
    make_run(runs, FINAL, splits=("test",))
    with pytest.raises(KeyError, match="No hay runs de 'mnv3_ft_cost' con predicciones de test y OOD"):
        cli(runs, tmp_path / "out", tmp_path / "x.csv")
    assert not (tmp_path / "out").exists()


def test_subset_underscore_and_missing_thresholds_are_skipped(tmp_path, capsys):
    runs = standard_runs(tmp_path)
    make_run(runs, "mnv3_head", subset=0.02)
    make_run(runs, "malexnet_aug", run_id="_smoke_malexnet_aug")
    no_thr = make_run(runs, "efficientnet_b0")
    (no_thr / "thresholds.json").unlink()
    groups = load_groups(runs)
    out = capsys.readouterr().out
    assert set(groups) == {B1, B2, FINAL}
    assert "subconjunto" in out and "sin umbrales" in out
    assert list(groups) == [B1, B2, FINAL]  # orden de la tabla de iteraciones


def test_refs_exclude_final_and_default_final_reads_model_card(tmp_path, monkeypatch):
    runs = standard_runs(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(paths, "models_dir", lambda: models)
    assert U.default_final() == U.DEFAULT_FINAL
    save_json(models / "best.json", {"name": B2})
    assert U.default_final() == B2
    out = tmp_path / "out"
    code = U.main(["--allow-test", "--runs-dir", str(runs), "--out", str(out),
                   "--iterations", str(tmp_path / "x.csv"), "--refs", B1, B2])
    assert code == 0
    csv = pd.read_csv(out / "incertidumbre.csv")
    deltas = csv[csv["kind"] == "delta"]
    assert set(deltas["group"]) == {B2} and set(deltas["reference"]) == {B1}
    assert f"**B2** ({B2}) — modelo final" in (out / "incertidumbre.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Datos reales (solo si existen las predicciones congeladas)
# ---------------------------------------------------------------------------

REAL_RUNS = paths.runs_dir()
REAL_ITER = paths.results_dir() / "iteraciones.csv"
REAL_UNC = paths.results_dir() / "incertidumbre.csv"
HAS_REAL = (REAL_ITER.is_file()
            and any(REAL_RUNS.glob(f"{FINAL}_s*/test_preds.npz"))
            and any(REAL_RUNS.glob(f"{B1}_s*/test_preds.npz")))


@pytest.mark.data
@pytest.mark.skipif(not HAS_REAL, reason="sin runs/*/test_preds.npz o results/iteraciones.csv")
def test_real_runs_point_estimates_match_iteraciones(tmp_path):
    protected = {name: snapshot(REPO / name) for name in ("models", "runs", "results")}
    out = tmp_path / "out"
    code = U.main(["--allow-test", "--out", str(out), "--n-boot", "5", "--final", FINAL])
    assert code == 0
    for name, snap in protected.items():
        assert snapshot(REPO / name) == snap, f"se modificó {name}/ del repo"
    table = pd.read_csv(out / "incertidumbre.csv")
    assert (table["n_boot"] == 5).all()
    it = pd.read_csv(REAL_ITER).set_index("group")
    for group in (FINAL, B1):
        for split in ("test", "ood"):
            for metric in ("f1_macro", "rate_busy_as_free", "value_comercial"):
                r = row(table, "metric", group, split, metric)
                assert r["estimate"] == pytest.approx(it.at[group, f"{split}_{metric}"], rel=1e-9), \
                    (group, split, metric)
    checked = table["matches_iteraciones"].dropna()
    assert len(checked) > 0 and checked.astype(bool).all()
    # Las estimaciones puntuales no dependen de n_boot: iguales a las de results/incertidumbre.csv
    if REAL_UNC.is_file():
        saved = pd.read_csv(REAL_UNC)
        key = ["kind", "group", "reference", "split", "metric"]
        merged = table.merge(saved, on=key, suffixes=("", "_saved"))
        assert len(merged) == len(saved) == len(table)
        np.testing.assert_allclose(merged["estimate"], merged["estimate_saved"], rtol=1e-9)
