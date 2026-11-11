"""Pruebas de evaluate.py: guarda de test/OOD, re-evaluación en val y umbrales fijos del run.

Nunca se ejecuta ``--allow-test``. Las pruebas de integración entrenan un run diminuto
(B2, 0.5 % de los datos, 1 época) en una carpeta temporal y re-evalúan en una submuestra
de val (``load_split`` se reemplaza por un registrador que submuestrea).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cupo import evaluate as E  # noqa: E402
from cupo import metrics as M  # noqa: E402
from cupo.runs import load_npz, save_json  # noqa: E402

from run_helpers import (  # noqa: E402
    HAS_DATA, PYTHON, REPO, file_hash, install_split_recorders, load_strict_json,
    load_train_classical, tiny_classical_cfg, tiny_torch_cfg,
)

needs_data = pytest.mark.skipif(not HAS_DATA, reason="dataset no disponible en DATA_ROOT")
TRAIN_ARTIFACTS = ["val_preds.npz", "metrics_val.json", "value_val.json", "thresholds.json",
                   "summary.json", "history.csv", "config.yaml", "best.pt"]
EVAL_FRAC = 0.01


# ================================================================ unitarias

def test_probs_from_logits_is_softmax_class_one():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, size=(50, 2)).astype(np.float32)
    for temp in (1.0, 0.5, 2.5):
        ref = torch.softmax(torch.from_numpy(logits).double() / temp, dim=1)[:, 1].numpy()
        np.testing.assert_allclose(E.probs_from_logits(logits, temp), ref, rtol=1e-10, atol=1e-12)


def test_probs_from_logits_stable_with_extreme_logits():
    p = E.probs_from_logits(np.array([[0.0, 1e4], [1e4, 0.0], [0.0, 0.0]]))
    assert np.isfinite(p).all()
    assert p[0] == pytest.approx(1.0) and p[1] == pytest.approx(0.0, abs=1e-20) and p[2] == 0.5


def test_classical_logits_are_zero_and_decision():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int)
    pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression())]).fit(X, y)
    logits = E.classical_logits({"pipeline": pipe}, X)
    assert logits.shape == (200, 2) and logits.dtype == np.float32
    assert (logits[:, 0] == 0).all()
    np.testing.assert_allclose(logits[:, 1], pipe.decision_function(X), rtol=1e-5)
    np.testing.assert_allclose(E.probs_from_logits(logits), pipe.predict_proba(X)[:, 1], atol=1e-6)


def test_scenario_names_primary_first():
    assert E.scenario_names("comercial") == ["comercial", "uvg"]
    assert E.scenario_names("uvg") == ["uvg", "comercial"]
    with pytest.raises(ValueError):
        E.scenario_names("otro")


def test_output_tag():
    assert E.output_tag("val") == "val_eval"
    assert E.output_tag("test") == "test"
    assert E.output_tag("ood") == "ood"


@pytest.mark.parametrize("split", ["test", "ood"])
def test_parse_args_blocks_protected_without_flag(split):
    with pytest.raises(SystemExit) as exc:
        E.parse_args(["--run", "x", "--split", split])
    assert exc.value.code != 0


def test_value_report_uses_given_thresholds(com, uvg, good_scores):
    y, p = good_scores
    y, p = y[:2000], p[:2000]
    thr = {"comercial": {"t_L": 0.1, "t_H": 0.9}, "uvg": {"t_L": 0.5, "t_H": 0.5}}
    rep = E.value_report(y, p, thr, {"comercial": com, "uvg": uvg}, n_boot=20)
    from cupo.value_model import compute_value

    for name, cfg in (("comercial", com), ("uvg", uvg)):
        t_L, t_H = thr[name]["t_L"], thr[name]["t_H"]
        rates = M.three_state_rates(y, p, t_L, t_H)
        assert (rep[name]["t_L"], rep[name]["t_H"]) == (t_L, t_H)
        assert rep[name]["objective_value"] == pytest.approx(compute_value(rates, cfg)["objective"])
    assert rep["uvg"]["rates"]["verify_rate"] == 0.0


# ============================================================== guarda CLI

@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """Run diminuto de B2 entrenado en una carpeta temporal."""
    if not HAS_DATA:
        pytest.skip("dataset no disponible en DATA_ROOT")
    from cupo.train import run_training

    root = tmp_path_factory.mktemp("runs_eval")
    mp = pytest.MonkeyPatch()
    try:
        rec, _ = install_split_recorders(mp)
        run_dir = run_training(tiny_torch_cfg(subset=0.005), run_name="b2_tiny", runs_root=root)
        assert rec.names == {"train", "val"}
    finally:
        mp.undo()
    return Path(run_dir)


@pytest.fixture(scope="module")
def tiny_classical_run(tmp_path_factory):
    if not HAS_DATA:
        pytest.skip("dataset no disponible en DATA_ROOT")
    mod = load_train_classical()
    root = tmp_path_factory.mktemp("runs_eval_b1")
    mp = pytest.MonkeyPatch()
    try:
        install_split_recorders(mp)
        run_dir = mod.run(tiny_classical_cfg(subset=0.01), run_name="b1_tiny", runs_root=root)
    finally:
        mp.undo()
    return Path(run_dir)


def _listing(d: Path) -> dict[str, str]:
    return {p.name: file_hash(p) for p in sorted(d.iterdir()) if p.is_file()}


@pytest.mark.data
@needs_data
@pytest.mark.parametrize("split", ["test", "ood"])
def test_cli_protected_split_fails_without_files(tiny_run, split):
    before = _listing(tiny_run)
    proc = subprocess.run([PYTHON, "-m", "cupo.evaluate", "--run", str(tiny_run), "--split", split],
                          cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=300)
    assert proc.returncode != 0
    assert "--allow-test" in proc.stderr
    assert _listing(tiny_run) == before
    assert not list(tiny_run.glob(f"*{split}*"))


@pytest.mark.parametrize("split", ["test", "ood"])
def test_main_protected_split_never_loads_data(monkeypatch, split):
    import cupo.data

    calls = []
    monkeypatch.setattr(cupo.data, "load_split", lambda *a, **k: calls.append(a) or None)
    monkeypatch.setattr(E, "load_run", lambda *a, **k: calls.append("load_run"))
    with pytest.raises(SystemExit) as exc:
        E.main(["--run", "x", "--split", split])
    assert exc.value.code != 0
    assert calls == []


# ========================================================= evaluación en val

def _eval_val(monkeypatch, run_dir: Path, extra_args=()):
    rec, _ = install_split_recorders(monkeypatch, frac=EVAL_FRAC)
    code = E.main(["--run", str(run_dir), "--split", "val", "--num-workers", "0",
                   "--n-boot", "10", *extra_args])
    assert code == 0
    assert rec.names == {"val"}
    return rec


@pytest.mark.data
@needs_data
def test_eval_val_writes_val_eval_and_keeps_training_artifacts(monkeypatch, tiny_run):
    before = {f: file_hash(tiny_run / f) for f in TRAIN_ARTIFACTS}
    _eval_val(monkeypatch, tiny_run)
    after = {f: file_hash(tiny_run / f) for f in TRAIN_ARTIFACTS}
    assert after == before
    for f in ("val_eval_preds.npz", "metrics_val_eval.json", "value_val_eval.json"):
        assert (tiny_run / f).is_file(), f
    m = load_strict_json(tiny_run / "metrics_val_eval.json")
    assert m["split"] == "val" and m["tag"] == "val_eval"
    assert m["temperature"] == 1.0 and m["calibrated"] is False and m["model_kind"] == "torch"
    preds = load_npz(tiny_run / "val_eval_preds.npz")
    assert {"y", "p", "logits", "camera", "weather", "date", "time", "slot_id", "path"} <= set(preds)
    assert m["n"] == preds["y"].size
    for key in ("binary", "ci95", "three_state", "thresholds", "value", "value_ci95", "breakdown"):
        assert key in m
    assert set(m["breakdown"]) == {"camera", "weather"}


@pytest.mark.data
@needs_data
def test_eval_val_uses_run_thresholds_without_reoptimizing(monkeypatch, tiny_run, tmp_path):
    run_copy = tmp_path / tiny_run.name
    shutil.copytree(tiny_run, run_copy)
    fixed = {"comercial": {"t_L": 0.13, "t_H": 0.87}, "uvg": {"t_L": 0.42, "t_H": 0.58}}
    save_json(run_copy / "thresholds.json", fixed)

    def forbidden(*a, **k):
        raise AssertionError("evaluate no debe re-optimizar umbrales")

    monkeypatch.setattr(E, "optimize", forbidden)
    monkeypatch.setattr(E, "choose_thresholds", forbidden)
    import cupo.thresholds

    monkeypatch.setattr(cupo.thresholds, "optimize", forbidden)
    _eval_val(monkeypatch, run_copy)
    m = load_strict_json(run_copy / "metrics_val_eval.json")
    v = load_strict_json(run_copy / "value_val_eval.json")
    assert m["thresholds"] == fixed
    for scen, thr in fixed.items():
        assert (v[scen]["t_L"], v[scen]["t_H"]) == (thr["t_L"], thr["t_H"])
    preds = load_npz(run_copy / "val_eval_preds.npz")
    rates = M.three_state_rates(preds["y"], preds["p"], 0.13, 0.87)
    for k, val in rates.items():
        assert m["three_state"]["comercial"][k] == pytest.approx(val)


@pytest.mark.data
@needs_data
def test_eval_val_applies_calibration_temperature(monkeypatch, tiny_run, tmp_path):
    run_copy = tmp_path / tiny_run.name
    shutil.copytree(tiny_run, run_copy)
    calib_thr = {"comercial": {"t_L": 0.2, "t_H": 0.8}, "uvg": {"t_L": 0.3, "t_H": 0.7}}
    save_json(run_copy / "calibration.json", {"temperature": 2.0, "thresholds": calib_thr})
    _eval_val(monkeypatch, run_copy)
    preds = load_npz(run_copy / "val_eval_preds.npz")
    ref = torch.softmax(torch.from_numpy(preds["logits"]).double() / 2.0, dim=1)[:, 1].numpy()
    np.testing.assert_allclose(preds["p"], ref, atol=1e-9)
    m = load_strict_json(run_copy / "metrics_val_eval.json")
    assert m["temperature"] == 2.0 and m["calibrated"] is True
    assert m["thresholds"] == calib_thr


@pytest.mark.data
@needs_data
def test_eval_val_logits_match_training_predictions(monkeypatch, tiny_run, tmp_path):
    """El modelo cargado de best.pt reproduce las probabilidades guardadas al entrenar."""
    run_copy = tmp_path / tiny_run.name
    shutil.copytree(tiny_run, run_copy)
    install_split_recorders(monkeypatch, frac=None)
    import cupo.data

    train_preds = load_npz(run_copy / "val_preds.npz")
    paths_set = set(train_preds["path"].tolist())
    original = cupo.data.load_split

    def only_train_val_rows(name, allow_test=False):
        df = original(name, allow_test=allow_test)
        return df[df["path"].isin(paths_set)].reset_index(drop=True)

    monkeypatch.setattr(cupo.data, "load_split", only_train_val_rows)
    assert E.main(["--run", str(run_copy), "--split", "val", "--num-workers", "0", "--n-boot", "5"]) == 0
    ev = load_npz(run_copy / "val_eval_preds.npz")
    assert ev["path"].tolist() == train_preds["path"].tolist()
    np.testing.assert_allclose(ev["p"], train_preds["p"], atol=5e-3)


@pytest.mark.data
@needs_data
def test_eval_val_classical_run(monkeypatch, tiny_classical_run):
    before = {f: file_hash(tiny_classical_run / f) for f in
              ("val_preds.npz", "metrics_val.json", "value_val.json", "thresholds.json", "summary.json")}
    _eval_val(monkeypatch, tiny_classical_run)
    after = {f: file_hash(tiny_classical_run / f) for f in before}
    assert after == before
    m = load_strict_json(tiny_classical_run / "metrics_val_eval.json")
    assert m["model_kind"] == "classical"
    preds = load_npz(tiny_classical_run / "val_eval_preds.npz")
    assert (preds["logits"][:, 0] == 0).all()
    np.testing.assert_allclose(preds["p"], E.probs_from_logits(preds["logits"]), atol=1e-12)
    assert m["thresholds"] == load_strict_json(tiny_classical_run / "thresholds.json")
