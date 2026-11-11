"""Pruebas de calibrate.py: temperature scaling en val, umbrales re-optimizados y uso en load_run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cupo import calibrate as C  # noqa: E402
from cupo.evaluate import probs_from_logits  # noqa: E402
from cupo.runs import load_run, save_json, save_npz, save_yaml  # noqa: E402
from cupo.thresholds import optimize  # noqa: E402

from run_helpers import install_split_recorders, load_strict_json  # noqa: E402


def _synthetic_logits(n: int, scale: float, seed: int = 0):
    """Logits de un clasificador calibrado (z ~ logit verdadero) multiplicados por ``scale``.

    Con ``scale`` > 1 el modelo queda sobreconfiado y la T óptima es ≈ ``scale``.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-z)))
    logits = np.stack([np.zeros(n), scale * z], axis=1).astype(np.float32)
    return logits, y


def _grid_temperature(logits, y):
    temps = np.linspace(0.2, 8.0, 3901)
    nlls = [C.nll(probs_from_logits(logits, t), y) for t in temps]
    return float(temps[int(np.argmin(nlls))])


def test_overconfident_logits_give_t_above_one():
    logits, y = _synthetic_logits(20_000, scale=3.0)
    t = C.fit_temperature(logits, y)
    assert t > 1.0
    assert t == pytest.approx(3.0, rel=0.1)
    assert t == pytest.approx(_grid_temperature(logits, y), abs=0.01)


def test_underconfident_logits_give_t_below_one():
    logits, y = _synthetic_logits(20_000, scale=0.4, seed=1)
    t = C.fit_temperature(logits, y)
    assert t < 1.0
    assert t == pytest.approx(_grid_temperature(logits, y), abs=0.01)


def test_calibrated_logits_give_t_near_one():
    logits, y = _synthetic_logits(20_000, scale=1.0, seed=2)
    assert C.fit_temperature(logits, y) == pytest.approx(1.0, abs=0.05)


def test_temperature_reduces_nll():
    logits, y = _synthetic_logits(5_000, scale=4.0, seed=3)
    t = C.fit_temperature(logits, y)
    assert C.nll(probs_from_logits(logits, t), y) < C.nll(probs_from_logits(logits, 1.0), y)


def test_probs_are_softmax_over_scaled_logits():
    logits, _ = _synthetic_logits(100, scale=2.0)
    logits[:, 0] = np.linspace(-1, 1, 100)  # clase 0 distinta de cero
    for t in (0.7, 1.0, 2.3):
        ref = torch.softmax(torch.from_numpy(logits).double() / t, dim=1)[:, 1].numpy()
        np.testing.assert_allclose(probs_from_logits(logits, t), ref, atol=1e-12)


def test_nll_and_ece_by_hand():
    p = np.array([0.9, 0.2, 0.6, 0.4])
    y = np.array([1, 0, 0, 1])
    expected = -(np.log(0.9) + np.log(0.8) + np.log(0.4) + np.log(0.4)) / 4
    assert C.nll(p, y) == pytest.approx(expected)
    # ECE perfecto: confianza 1 y siempre acierta
    assert C.ece(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)
    # confianza 0.8 en los 4 casos y 2 aciertos: |0.8 - 0.5| = 0.3
    assert C.ece(np.array([0.8, 0.8, 0.2, 0.2]), np.array([1, 0, 0, 1])) == pytest.approx(0.3)


# ------------------------------------------------------- calibrate_run sobre un run

def _fake_run(run_dir: Path, scale: float = 3.0) -> Path:
    from cupo.models import build_model

    run_dir.mkdir(parents=True)
    logits, y = _synthetic_logits(4_000, scale=scale, seed=5)
    n = y.size
    save_npz(run_dir / "val_preds.npz", y=y, p=probs_from_logits(logits), logits=logits,
             camera=np.array(["1"] * n), weather=np.array(["SUNNY"] * n),
             date=np.array(["2015-11-20"] * n), path=np.array([f"x{i}.jpg" for i in range(n)]))
    save_yaml(run_dir / "config.yaml", {"name": "fake", "select": {"scenario": "comercial"}})
    save_json(run_dir / "thresholds.json", {"comercial": {"t_L": 0.5, "t_H": 0.5},
                                            "uvg": {"t_L": 0.5, "t_H": 0.5}})
    model = build_model("malexnet", pretrained=False, input_size=128)
    torch.save({"state_dict": model.state_dict(), "model_cfg": {"name": "malexnet"},
                "input_size": 128, "epoch": 1}, run_dir / "best.pt")
    return run_dir


@pytest.fixture
def calibrated(tmp_path, monkeypatch):
    rec, inner = install_split_recorders(monkeypatch)
    run_dir = _fake_run(tmp_path / "fake_run")
    val_before = (run_dir / "val_preds.npz").read_bytes()
    thr_before = (run_dir / "thresholds.json").read_bytes()
    res = C.calibrate_run(str(run_dir), n_boot=10)
    return {"run_dir": run_dir, "res": res, "rec": rec, "inner": inner,
            "val_before": val_before, "thr_before": thr_before}


def test_calibrate_run_writes_calibration_json(calibrated):
    run_dir, res = calibrated["run_dir"], calibrated["res"]
    data = load_strict_json(run_dir / "calibration.json")
    for key in ("temperature", "thresholds", "nll_before", "nll_after", "ece_before", "ece_after",
                "value_val", "fitted_on"):
        assert key in data
    assert data["fitted_on"] == "val"
    assert data["temperature"] == pytest.approx(res["temperature"])
    assert data["temperature"] > 1.0
    assert data["nll_after"] <= data["nll_before"]
    assert set(data["thresholds"]) == {"comercial", "uvg"}
    for thr in data["thresholds"].values():
        assert 0.0 <= thr["t_L"] <= thr["t_H"] <= 1.0


def test_calibrate_thresholds_are_reoptimized_on_calibrated_val(calibrated):
    from cupo.runs import load_npz
    from cupo.value_model import load_scenario

    run_dir = calibrated["run_dir"]
    data = load_strict_json(run_dir / "calibration.json")
    preds = load_npz(run_dir / "val_preds.npz")
    p_cal = probs_from_logits(preds["logits"], data["temperature"])
    for scen in ("comercial", "uvg"):
        best = optimize(preds["y"], p_cal, load_scenario(scen))
        assert data["thresholds"][scen] == {"t_L": pytest.approx(best["t_L"]),
                                            "t_H": pytest.approx(best["t_H"])}
        assert data["value_val"][scen]["objective_value"] == pytest.approx(best["value"])


def test_calibrate_does_not_modify_training_artifacts_nor_read_splits(calibrated):
    run_dir = calibrated["run_dir"]
    assert (run_dir / "val_preds.npz").read_bytes() == calibrated["val_before"]
    assert (run_dir / "thresholds.json").read_bytes() == calibrated["thr_before"]
    # calibrate solo usa los logits guardados de val: no carga ninguna partición
    assert calibrated["rec"].calls == []
    assert calibrated["inner"].names == []


def test_load_run_applies_calibration(calibrated):
    run_dir = calibrated["run_dir"]
    data = load_strict_json(run_dir / "calibration.json")
    run = load_run(run_dir)
    assert run.calibrated is True
    assert run.temperature == pytest.approx(data["temperature"])
    assert run.thresholds == data["thresholds"]
    assert run.thresholds != load_strict_json(run_dir / "thresholds.json")


def test_calibrate_without_logits_raises(tmp_path):
    run_dir = tmp_path / "sin_logits"
    run_dir.mkdir()
    save_npz(run_dir / "val_preds.npz", y=[0, 1], p=[0.2, 0.8])
    save_yaml(run_dir / "config.yaml", {"name": "x"})
    with pytest.raises(KeyError):
        C.calibrate_run(str(run_dir), n_boot=5)


def test_calibrate_cli(tmp_path, monkeypatch, capsys):
    install_split_recorders(monkeypatch)
    run_dir = _fake_run(tmp_path / "cli_run", scale=2.0)
    assert C.main(["--run", str(run_dir), "--n-boot", "5"]) == 0
    assert "T = " in capsys.readouterr().out
    assert load_strict_json(run_dir / "calibration.json")["temperature"] > 1.0
