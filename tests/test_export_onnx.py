"""Pruebas de export_onnx.py: equivalencia ONNX/PyTorch, best.json y rechazos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from PIL import Image  # noqa: E402

from cupo import export_onnx as E  # noqa: E402
from cupo import paths  # noqa: E402
from cupo.evaluate import probs_from_logits  # noqa: E402
from cupo.runs import load_json, load_npz, load_run  # noqa: E402
from cupo.transforms import IMAGENET_MEAN, IMAGENET_STD, preprocess_numpy  # noqa: E402

from export_helpers import assert_no_local_paths, fake_classical_run, torch_run_copy  # noqa: E402
from run_helpers import HAS_DATA, SplitRecorder, load_strict_json  # noqa: E402

pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not HAS_DATA, reason="requiere el dataset en DATA_ROOT")]

CARD_KEYS = {"name", "run_id", "model", "onnx_file", "input_size", "input_name", "output_name",
             "classes", "preprocessing", "opset", "temperature", "calibrated", "thresholds",
             "select_scenario", "val_metrics_summary", "onnx_check"}
PREPROCESSING_KEYS = {"color", "size", "resize", "scale", "mean", "std", "layout", "function"}


def _export(run_dir: Path, out: Path, *extra: str):
    """Exporta en el mismo proceso registrando las particiones pedidas."""
    import cupo.data

    with pytest.MonkeyPatch.context() as mp:
        rec = SplitRecorder(cupo.data.load_split)
        mp.setattr(cupo.data, "load_split", rec)
        rc = E.main(["--run", str(run_dir), "--out", str(out), *extra])
    return rc, rec


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Exporta una vez mAlexNet (sin calibrar) y MobileNetV3 (calibrado) en carpetas temporales."""
    out = {}
    for kind in ("malexnet", "mnv3_calibrated"):
        root = tmp_path_factory.mktemp(f"onnx_{kind}")
        run_dir = torch_run_copy(kind, root / "runs")
        onnx_path = root / "models" / "best.onnx"
        rc, rec = _export(run_dir, onnx_path)
        out[kind] = {"root": root, "run_dir": run_dir, "onnx": onnx_path, "rc": rc, "rec": rec,
                     "card_path": onnx_path.with_suffix(".json")}
    return out


@pytest.fixture(params=["malexnet", "mnv3_calibrated"])
def case(request, exported):
    return exported[request.param]


def _val_batch(input_size: int, n: int) -> np.ndarray:
    from cupo.data import load_split

    df = load_split("val").sample(n=n, random_state=123)
    arrs = []
    for p in df["path"]:
        with Image.open(p) as im:
            arrs.append(preprocess_numpy(im, input_size))
    return np.concatenate(arrs)


def _relocate(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    parts = Path(path.replace("\\", "/")).parts
    return paths.data_root().joinpath(*parts[parts.index(paths.EXT_DIRNAME):])


# ---------------------------------------------------------------------------
# Exportación y equivalencia con PyTorch
# ---------------------------------------------------------------------------

def test_export_returns_zero_and_writes_onnx_and_card(case):
    assert case["rc"] == 0
    assert case["onnx"].is_file() and case["card_path"].is_file()
    assert sorted(p.name for p in case["onnx"].parent.iterdir()) == ["best.json", "best.onnx"]


def test_onnx_graph_has_dynamic_batch_and_opset_17(case):
    model = onnx.load(str(case["onnx"]))
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    assert opset >= 17
    (inp,) = model.graph.input
    (out,) = model.graph.output
    assert inp.name == "input" and out.name == "logits"
    in_dims = inp.type.tensor_type.shape.dim
    assert in_dims[0].dim_param and not in_dims[0].dim_value  # eje 0 simbólico
    assert out.type.tensor_type.shape.dim[0].dim_param
    assert [d.dim_value for d in in_dims[1:]] == [3, load_json(case["card_path"])["input_size"],
                                                  load_json(case["card_path"])["input_size"]]


def test_onnx_matches_torch_batch_1_and_batch_n(case):
    run = load_run(case["run_dir"], device="cpu")
    batch = _val_batch(run.input_size, 6)
    with torch.no_grad():
        ref = run.model.float().eval()(torch.from_numpy(batch)).numpy()
    sess = ort.InferenceSession(str(case["onnx"]), providers=["CPUExecutionProvider"])
    got_n = sess.run(["logits"], {"input": batch})[0]
    got_1 = np.concatenate([sess.run(["logits"], {"input": batch[i:i + 1]})[0] for i in range(len(batch))])
    assert got_n.shape == (6, 2)
    assert np.abs(got_n - ref).max() < 1e-3
    assert np.abs(got_1 - ref).max() < 1e-3
    # Otro tamaño de batch distinto al dummy de exportación (2)
    got_3 = sess.run(["logits"], {"input": batch[:3]})[0]
    assert np.abs(got_3 - ref[:3]).max() < 1e-3


def test_card_records_passing_check(case):
    check = load_json(case["card_path"])["onnx_check"]
    assert check["ok"] is True
    assert check["tolerance"] == pytest.approx(1e-3)
    assert check["max_abs_diff_logits"] < 1e-3
    assert check["n"] == E.CHECK_N


def test_export_only_reads_val(case):
    assert case["rec"].names == {"val"}
    assert all(not allow for _, allow in case["rec"].calls)


# ---------------------------------------------------------------------------
# best.json (contrato con la app)
# ---------------------------------------------------------------------------

def test_card_contract_fields(case):
    card = load_strict_json(case["card_path"])
    assert CARD_KEYS <= set(card)
    run = load_run(case["run_dir"], device="cpu")
    assert card["run_id"] == case["run_dir"].name
    assert card["name"] == run.config["name"]
    assert card["onnx_file"] == "best.onnx"
    assert card["input_size"] == run.input_size
    assert card["input_name"] == "input" and card["output_name"] == "logits"
    assert card["classes"] == ["libre", "ocupado"]
    assert card["opset"] >= 17
    assert set(card["thresholds"]) == {"comercial", "uvg"}
    for thr in card["thresholds"].values():
        assert set(thr) == {"t_L", "t_H"}
        assert 0.0 <= thr["t_L"] <= thr["t_H"] <= 1.0
    summary = card["val_metrics_summary"]
    assert {"source", "binary_t05", "value"} <= set(summary)
    assert set(summary["value"]) == {"comercial", "uvg"}
    for v in summary["value"].values():
        assert {"objective_name", "objective_value", "ci95", "verify_rate"} <= set(v)
        assert v["objective_value"] is not None


def test_card_names_match_onnx_session(case):
    card = load_json(case["card_path"])
    sess = ort.InferenceSession(str(case["onnx"]), providers=["CPUExecutionProvider"])
    assert sess.get_inputs()[0].name == card["input_name"]
    assert sess.get_outputs()[0].name == card["output_name"]


def test_card_preprocessing_is_complete_and_reproduces_preprocess_numpy(case):
    card = load_json(case["card_path"])
    pre = card["preprocessing"]
    assert PREPROCESSING_KEYS <= set(pre)
    size = card["input_size"]
    assert pre["size"] == [size, size]
    assert pre["color"] == "RGB" and pre["layout"] == "NCHW" and pre["resize"] == "bilinear"
    assert pre["mean"] == pytest.approx(list(IMAGENET_MEAN))
    assert pre["std"] == pytest.approx(list(IMAGENET_STD))
    # Una app que solo lea best.json obtiene el mismo tensor que preprocess_numpy
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (150, 150, 3), dtype=np.uint8))
    arr = np.asarray(img.convert(pre["color"]).resize(tuple(pre["size"]), Image.Resampling.BILINEAR),
                     dtype=np.float32) / 255.0
    arr = (arr - np.array(pre["mean"], dtype=np.float32)) / np.array(pre["std"], dtype=np.float32)
    manual = arr.transpose(2, 0, 1)[None]
    np.testing.assert_allclose(manual, preprocess_numpy(img, size), atol=1e-6)


def test_card_has_no_local_paths(case):
    text = case["card_path"].read_text(encoding="utf-8")
    assert_no_local_paths(text, case["root"], case["run_dir"])


def test_uncalibrated_run_uses_t1_and_thresholds_json(exported):
    case = exported["malexnet"]
    card = load_json(case["card_path"])
    thr = load_json(case["run_dir"] / "thresholds.json")
    assert card["calibrated"] is False
    assert card["temperature"] == 1.0
    for s in ("comercial", "uvg"):
        assert card["thresholds"][s]["t_L"] == pytest.approx(thr[s]["t_L"])
        assert card["thresholds"][s]["t_H"] == pytest.approx(thr[s]["t_H"])
    assert card["val_metrics_summary"]["source"] == "value_val.json"
    value = load_json(case["run_dir"] / "value_val.json")
    for s in ("comercial", "uvg"):
        assert card["val_metrics_summary"]["value"][s]["objective_value"] == pytest.approx(
            value[s]["objective_value"])


def test_calibrated_run_uses_temperature_and_thresholds_of_calibration(exported):
    case = exported["mnv3_calibrated"]
    card = load_json(case["card_path"])
    calib = load_json(case["run_dir"] / "calibration.json")
    assert card["calibrated"] is True
    assert card["temperature"] == pytest.approx(calib["temperature"])
    assert card["temperature"] != 1.0
    for s in ("comercial", "uvg"):
        assert card["thresholds"][s] == pytest.approx(calib["thresholds"][s])
    summary = card["val_metrics_summary"]
    assert summary["source"] == "calibration.json"
    for s in ("comercial", "uvg"):
        assert summary["value"][s]["objective_value"] == pytest.approx(calib["value_val"][s]["objective_value"])
        assert summary["value"][s]["verify_rate"] == pytest.approx(calib["value_val"][s]["verify_rate"])


def test_onnx_plus_card_reproduce_val_preds(case):
    """p de ONNX (best.json + preprocess_numpy) ≈ p de val_preds.npz con la T del run."""
    card = load_json(case["card_path"])
    preds = load_npz(case["run_dir"] / "val_preds.npz")
    idx = np.random.default_rng(1).choice(len(preds["y"]), size=min(40, len(preds["y"])), replace=False)
    batch = []
    for i in idx:
        with Image.open(_relocate(str(preds["path"][i]))) as im:
            batch.append(preprocess_numpy(im, card["input_size"]))
    sess = ort.InferenceSession(str(case["onnx"]), providers=["CPUExecutionProvider"])
    logits = sess.run([card["output_name"]], {card["input_name"]: np.concatenate(batch)})[0]
    z = (logits[:, 1] - logits[:, 0]) / card["temperature"]
    p_onnx = 1.0 / (1.0 + np.exp(-z))
    if "logits" in preds:
        p_ref = probs_from_logits(preds["logits"][idx], card["temperature"])
    else:
        assert card["temperature"] == 1.0
        p_ref = preds["p"][idx]
    assert np.abs(p_onnx - p_ref).max() < 0.02


# ---------------------------------------------------------------------------
# Rechazos y errores
# ---------------------------------------------------------------------------

def test_classical_run_fails_with_clear_error(tmp_path, capsys):
    run_dir = fake_classical_run(tmp_path / "runs" / "b1_hog_lr_s42_x")
    out = tmp_path / "models" / "best.onnx"
    with pytest.raises(ValueError, match="clásico"):
        E.load_exportable_run(run_dir)
    rc = E.main(["--run", str(run_dir), "--out", str(out)])
    assert rc != 0
    err = capsys.readouterr().err
    assert "clásico" in err and "ONNX" in err
    assert not out.exists() and not out.with_suffix(".json").exists()


@pytest.mark.parametrize("opset", ["16", "11"])
def test_opset_below_17_is_rejected(tmp_path, capsys, opset):
    out = tmp_path / "best.onnx"
    with pytest.raises(SystemExit) as exc:
        E.main(["--run", "no_importa", "--out", str(out), "--opset", opset])
    assert exc.value.code != 0
    assert "--opset" in capsys.readouterr().err
    assert not out.exists()


def test_missing_run_fails(tmp_path, capsys):
    rc = E.main(["--run", str(tmp_path / "no_existe"), "--out", str(tmp_path / "best.onnx")])
    assert rc != 0
    assert "Error" in capsys.readouterr().err


def test_run_without_scenario_thresholds_fails(tmp_path, exported):
    import shutil

    from cupo.runs import save_json

    src = exported["malexnet"]["run_dir"]
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    save_json(dst / "thresholds.json", {"comercial": {"t_L": 0.4, "t_H": 0.6}})
    with pytest.raises(ValueError, match="uvg"):
        E.load_exportable_run(dst)


def test_check_failure_returns_error_code(tmp_path, exported, monkeypatch):
    """Si ONNX y PyTorch no coinciden dentro de la tolerancia, termina con rc != 0."""
    monkeypatch.setattr(E, "CHECK_TOL", 0.0)
    out = tmp_path / "best.onnx"
    rc = E.main(["--run", str(exported["malexnet"]["run_dir"]), "--out", str(out), "--n-check", "2"])
    assert rc != 0
    card = out.with_suffix(".json")
    if card.exists():
        assert load_json(card)["onnx_check"]["ok"] is False


def test_default_out_is_models_dir():
    """Sin --out, el destino es models/best.onnx del repo (solo se revisa el parser)."""
    args = E.parse_args(["--run", "x"])
    assert Path(args.out) == paths.models_dir() / "best.onnx"
    assert args.opset == 17


def test_module_entrypoint_runs(exported, tmp_path):
    """``python -m cupo.export_onnx`` en un proceso aparte (cwd temporal)."""
    import os
    import subprocess

    from run_helpers import PYTHON, REPO

    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    out = tmp_path / "m" / "best.onnx"
    proc = subprocess.run([PYTHON, "-m", "cupo.export_onnx", "--run", str(exported["malexnet"]["run_dir"]),
                           "--out", str(out), "--n-check", "2"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert out.is_file() and out.with_suffix(".json").is_file()
