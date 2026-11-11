"""Pruebas de runs.py: identificadores, carpetas, JSON estricto, npz y carga de un run."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from cupo import paths
from cupo import runs as R

from run_helpers import load_strict_json


# ---------------------------------------------------------- identificadores

def test_make_run_id_format():
    now = datetime(2026, 11, 11, 9, 5)
    assert R.make_run_id("b2_malexnet", 42, now) == "b2_malexnet_s42_20261111-0905"
    assert R.make_run_id("k3", 7, datetime(2026, 1, 2, 23, 59)) == "k3_s7_20260102-2359"


def test_make_run_id_uses_current_time():
    rid = R.make_run_id("x", 1)
    stamp = rid.split("_s1_")[1]
    datetime.strptime(stamp, "%Y%m%d-%H%M")


def test_create_run_dir_collision_appends_suffix(tmp_path):
    a = R.create_run_dir("run_s1_20261111-1200", tmp_path)
    b = R.create_run_dir("run_s1_20261111-1200", tmp_path)
    c = R.create_run_dir("run_s1_20261111-1200", tmp_path)
    assert a.name == "run_s1_20261111-1200"
    assert b.name == "run_s1_20261111-1200-2"
    assert c.name == "run_s1_20261111-1200-3"
    assert all(p.is_dir() and p.is_absolute() for p in (a, b, c))
    assert a.parent == tmp_path.resolve()


def test_create_run_dir_creates_missing_root(tmp_path):
    root = tmp_path / "nueva" / "runs"
    d = R.create_run_dir("abc", root)
    assert d.is_dir() and d.parent == root.resolve()


# ------------------------------------------------------------- JSON y YAML

def test_save_json_nan_and_inf_become_null(tmp_path):
    data = {
        "nan": float("nan"),
        "np_inf": np.float32(np.inf),
        "neg_inf": -math.inf,
        "np_int": np.int64(3),
        "np_float": np.float64(0.25),
        "np_bool": np.bool_(True),
        "arr": np.array([1.0, np.nan, 2.0]),
        "nested": {"x": [np.nan, (1, np.float32(2.5))]},
        "path": Path("a/b"),
        5: "clave numérica",
    }
    out = tmp_path / "m.json"
    R.save_json(out, data)
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    got = load_strict_json(out)
    assert got["nan"] is None and got["np_inf"] is None and got["neg_inf"] is None
    assert got["np_int"] == 3 and isinstance(got["np_int"], int)
    assert got["np_float"] == 0.25
    assert got["np_bool"] is True
    assert got["arr"] == [1.0, None, 2.0]
    assert got["nested"] == {"x": [None, [1, 2.5]]}
    assert got["path"] == str(Path("a/b"))
    assert got["5"] == "clave numérica"
    assert R.load_json(out) == got


def test_save_json_keeps_utf8(tmp_path):
    out = tmp_path / "u.json"
    R.save_json(out, {"nota": "época, valor"})
    assert "época" in out.read_text(encoding="utf-8")


def test_yaml_roundtrip_with_numpy(tmp_path):
    out = tmp_path / "c.yaml"
    R.save_yaml(out, {"a": np.float64(1e-3), "b": [np.int32(2)], "c": {"d": "ñ"}})
    assert R.load_yaml(out) == {"a": 1e-3, "b": [2], "c": {"d": "ñ"}}


def test_load_yaml_empty_file(tmp_path):
    out = tmp_path / "vacio.yaml"
    out.write_text("", encoding="utf-8")
    assert R.load_yaml(out) == {}


def test_deep_merge_does_not_modify_inputs():
    base = {"a": {"b": 1, "c": [1, 2]}, "x": 1}
    over = {"a": {"b": 2}, "y": {"z": 3}}
    merged = R.deep_merge(base, over)
    assert merged == {"a": {"b": 2, "c": [1, 2]}, "x": 1, "y": {"z": 3}}
    assert base == {"a": {"b": 1, "c": [1, 2]}, "x": 1}
    merged["a"]["c"].append(9)
    assert base["a"]["c"] == [1, 2]


def test_npz_roundtrip_strings_without_pickle(tmp_path):
    out = tmp_path / "p.npz"
    R.save_npz(out, y=[0, 1], p=np.array([0.1, 0.9]),
               camera=np.array(["1", "A"], dtype=object), path=["x/y.jpg", "z.jpg"])
    got = R.load_npz(out)
    assert got["y"].tolist() == [0, 1]
    assert got["camera"].dtype.kind == "U" and got["camera"].tolist() == ["1", "A"]
    assert got["path"].tolist() == ["x/y.jpg", "z.jpg"]


# ------------------------------------------------------------ resolve_run_dir

def test_resolve_run_dir_absolute_relative_and_id(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    run = runs_root / "b2_s42_20261111-0000"
    run.mkdir(parents=True)
    assert R.resolve_run_dir(run) == run.resolve()
    assert R.resolve_run_dir(str(run)) == run.resolve()
    monkeypatch.chdir(tmp_path)
    assert R.resolve_run_dir("runs/b2_s42_20261111-0000") == run.resolve()
    # solo el run_id: se busca dentro de runs/ (cwd en otra carpeta)
    monkeypatch.setattr(paths, "runs_dir", lambda: runs_root)
    other = tmp_path / "otro"
    other.mkdir()
    monkeypatch.chdir(other)
    assert R.resolve_run_dir("b2_s42_20261111-0000") == run.resolve()


def test_resolve_run_dir_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        R.resolve_run_dir("no_existe_s1_20000101-0000")


# ------------------------------------------------------------------ load_run

torch = pytest.importorskip("torch")


def _fake_torch_run(run_dir: Path, thresholds: dict | None = None) -> Path:
    from cupo.models import build_model

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = build_model("malexnet", pretrained=False, input_size=128)
    torch.save({"state_dict": model.state_dict(),
                "model_cfg": {"name": "malexnet", "pretrained": False, "freeze": "none", "num_classes": 2},
                "input_size": 128, "epoch": 3}, run_dir / "best.pt")
    R.save_yaml(run_dir / "config.yaml", {"name": "fake", "select": {"scenario": "comercial"}})
    if thresholds is not None:
        R.save_json(run_dir / "thresholds.json", thresholds)
    return run_dir


def test_load_run_torch_without_calibration(tmp_path):
    thr = {"comercial": {"t_L": 0.2, "t_H": 0.7}, "uvg": {"t_L": 0.3, "t_H": 0.6}}
    run_dir = _fake_torch_run(tmp_path / "r1", thr)
    run = R.load_run(run_dir)
    assert run.kind == "torch"
    assert run.input_size == 128
    assert run.temperature == 1.0 and run.calibrated is False
    assert run.thresholds == thr
    assert run.run_id == "r1"
    assert not run.model.training
    # pesos idénticos al checkpoint
    ckpt = torch.load(run_dir / "best.pt", weights_only=True)
    for k, v in run.model.state_dict().items():
        assert torch.equal(v.cpu(), ckpt["state_dict"][k])
    with torch.no_grad():
        assert run.model(torch.zeros(1, 3, 128, 128)).shape == (1, 2)


def test_load_run_uses_calibration_temperature_and_thresholds(tmp_path):
    thr = {"comercial": {"t_L": 0.2, "t_H": 0.7}}
    run_dir = _fake_torch_run(tmp_path / "r2", thr)
    calib_thr = {"comercial": {"t_L": 0.05, "t_H": 0.95}, "uvg": {"t_L": 0.1, "t_H": 0.9}}
    R.save_json(run_dir / "calibration.json", {"temperature": 1.8, "thresholds": calib_thr})
    run = R.load_run(run_dir)
    assert run.temperature == pytest.approx(1.8)
    assert run.thresholds == calib_thr
    assert run.calibrated is True


def test_load_run_without_model_raises(tmp_path):
    d = tmp_path / "vacio"
    d.mkdir()
    R.save_yaml(d / "config.yaml", {"name": "x"})
    with pytest.raises(FileNotFoundError):
        R.load_run(d)


def test_load_run_classical(tmp_path):
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    d = tmp_path / "b1"
    d.mkdir()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = (X[:, 0] > 0).astype(int)
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())]).fit(X, y)
    joblib.dump({"pipeline": pipe, "hog": {}, "C": 1.0}, d / "model.joblib")
    R.save_yaml(d / "config.yaml", {"name": "b1"})
    R.save_json(d / "thresholds.json", {"comercial": {"t_L": 0.4, "t_H": 0.6}})
    run = R.load_run(d)
    assert run.kind == "classical"
    assert run.input_size is None
    assert set(run.model) == {"pipeline", "hog", "C"}
    assert run.thresholds == {"comercial": {"t_L": 0.4, "t_H": 0.6}}


def test_checkpoint_loads_with_weights_only(tmp_path):
    """best.pt debe cargarse con weights_only=True (sin objetos arbitrarios)."""
    run_dir = _fake_torch_run(tmp_path / "r3", {})
    ckpt = torch.load(run_dir / "best.pt", weights_only=True)
    assert set(ckpt) == {"state_dict", "model_cfg", "input_size", "epoch"}


def test_json_module_rejects_nan_by_default_check():
    """Comprobación del lector estricto usado en las pruebas."""
    with pytest.raises(ValueError):
        json.loads('{"a": NaN}', parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
