"""Pruebas de benchmark.py: fórmulas a mano, benchmark.json, coherencia con best.json y particiones."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from cupo import benchmark as B  # noqa: E402
from cupo import export_onnx as E  # noqa: E402
from cupo import paths  # noqa: E402
from cupo.runs import load_json, load_yaml  # noqa: E402

from export_helpers import assert_no_local_paths, torch_run_copy  # noqa: E402
from run_helpers import HAS_DATA, SplitRecorder, load_strict_json  # noqa: E402

N_SLOT, N_FRAME = 6, 2


# ---------------------------------------------------------------------------
# Unidades con valores calculados a mano
# ---------------------------------------------------------------------------

def test_latency_stats_by_hand():
    s = B.latency_stats([1.0, 2.0, 3.0, 4.0], batch=2)
    assert s["batch"] == 2 and s["n"] == 4
    assert s["p50_ms"] == pytest.approx(2.5)
    # percentil 95 lineal: 3 + 0.85 · (4 - 3)
    assert s["p95_ms"] == pytest.approx(3.85)
    assert s["mean_ms"] == pytest.approx(2.5)
    # 2 predicciones cada 2.5 ms → 800 pred/s
    assert s["throughput_pred_s"] == pytest.approx(800.0)


def test_cost_per_1000_by_hand():
    c = B.cost_per_1000(500.0, usd_per_hour=0.05, gtq_per_usd=7.7)
    # 1000 / 500 = 2 s de núcleo; 2 / 3600 · 0.05 USD; ×7.7 Q
    assert c["core_seconds_per_1000"] == pytest.approx(2.0)
    assert c["usd"] == pytest.approx(2.0 / 3600 * 0.05)
    assert c["q"] == pytest.approx(2.0 / 3600 * 0.05 * 7.7)


def test_volume_by_hand():
    v = B.volume(frame_mean_ms=100.0, preprocess_ms=2.0, slots=46, refresh_s=30.0)
    # (100 + 46·2) ms = 0.192 s por frame → 30 / 0.192 = 156.25 cámaras
    assert v["seconds_per_frame_1core"] == pytest.approx(0.192)
    assert v["cameras_per_core"] == 156
    assert v["cameras_per_core_at_70pct"] == 109          # floor(156.25 · 0.7 = 109.375)
    assert v["slots_per_core"] == 7187                    # floor(156.25 · 46 = 7187.5)


def test_scenario_volume_uses_value_configs():
    from cupo.value_model import load_scenario

    out = B.scenario_volume(frame_s=0.2)
    assert set(out) == {"comercial", "uvg"}
    for name, sc in out.items():
        prm = load_scenario(name)["params"]
        n_cams = int(prm["n_camaras"])
        refresh = float(prm["refresh_s"])
        assert sc["n_camaras"] == n_cams
        assert sc["core_utilization"] == pytest.approx(n_cams * 0.2 / refresh)
        assert sc["fits_1_core"] == (n_cams * 0.2 / refresh <= 0.7)
        assert sc["pred_per_s_needed"] == pytest.approx(int(prm["espacios"]) / refresh)


def test_make_frame_batch_repeats_patches():
    patches = np.arange(3 * 1 * 1 * 1, dtype=np.float32).reshape(3, 1, 1, 1)
    frame = B.make_frame_batch(patches, batch=7)
    assert frame.shape == (7, 1, 1, 1)
    assert frame[:, 0, 0, 0].tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert frame.flags["C_CONTIGUOUS"]


def test_time_calls_runs_warmup_then_n():
    calls = []
    times = B.time_calls(calls.append, n=4, warmup=3)
    assert calls == [0, 1, 2, 0, 1, 2, 3]
    assert len(times) == 4 and all(t >= 0 for t in times)


def test_default_out_is_models_dir():
    args = B.parse_args([])
    assert Path(args.out) == paths.models_dir()
    assert Path(args.onnx) == paths.models_dir() / "best.onnx"


# ---------------------------------------------------------------------------
# Corridas cortas (con dataset)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """mAlexNet exportado a ONNX en una carpeta temporal (run copiado ahí mismo)."""
    if not HAS_DATA:
        pytest.skip("requiere el dataset en DATA_ROOT")
    root = tmp_path_factory.mktemp("bench")
    run_dir = torch_run_copy("malexnet", root / "runs")
    onnx_path = root / "models" / "best.onnx"
    assert E.main(["--run", str(run_dir), "--out", str(onnx_path), "--n-check", "2"]) == 0
    return {"root": root, "run_dir": run_dir, "onnx": onnx_path}


@pytest.fixture(scope="module")
def bench(exported):
    """Benchmark corto (--n pequeño, sin GPU) registrando las particiones pedidas."""
    import cupo.data

    out = exported["root"] / "bench_out"
    with pytest.MonkeyPatch.context() as mp:
        rec = SplitRecorder(cupo.data.load_split)
        mp.setattr(cupo.data, "load_split", rec)
        rc = B.main(["--onnx", str(exported["onnx"]), "--run", str(exported["run_dir"]),
                     "--n", str(N_SLOT), "--n-frame", str(N_FRAME), "--out", str(out), "--no-gpu"])
    return {**exported, "rc": rc, "rec": rec, "out": out, "json": out / "benchmark.json"}


@pytest.mark.data
def test_benchmark_writes_json_in_out_and_run(bench):
    assert bench["rc"] == 0
    assert bench["json"].is_file()
    in_run = bench["run_dir"] / "benchmark.json"
    assert in_run.is_file()
    a, b = load_json(bench["json"]), load_json(in_run)
    assert a == b
    assert sorted(p.name for p in bench["out"].iterdir()) == ["benchmark.json"]


@pytest.mark.data
def test_benchmark_json_keys(bench):
    res = load_strict_json(bench["json"])
    for key in ("model", "input_size", "data", "warmup", "n_slot", "n_frame", "preprocess_ms_per_patch",
                "preprocess_basis", "cpu_1_thread", "cpu_all_threads", "gpu_torch",
                "costo_por_1000_pred_Q", "cost", "volume", "volume_by_scenario", "env"):
        assert key in res, key
    assert res["warmup"] == B.WARMUP
    assert res["n_slot"] == N_SLOT and res["n_frame"] == N_FRAME
    assert res["gpu_torch"] is None  # --no-gpu
    assert isinstance(res["preprocess_basis"], str) and res["preprocess_basis"]
    for block in ("cpu_1_thread", "cpu_all_threads"):
        for mode, batch, n in (("per_slot", 1, N_SLOT), ("per_frame", B.SLOTS_PER_FRAME, N_FRAME)):
            st = res[block][mode]
            assert st["batch"] == batch and st["n"] == n
            for k in ("p50_ms", "p95_ms", "mean_ms", "throughput_pred_s"):
                assert st[k] > 0, (block, mode, k)
            assert st["p95_ms"] >= st["p50_ms"]
            assert st["throughput_pred_s"] == pytest.approx(batch / (st["mean_ms"] / 1000))
    assert res["cpu_1_thread"]["threads"] == 1
    cost = res["cost"]
    assert {"core_seconds_per_1000", "usd", "q", "basis", "pipeline_pred_s", "model_only",
            "estimate_note", "cpu_server_usd_per_hour", "gtq_per_usd"} <= set(cost)
    assert {"core_seconds_per_1000", "usd", "q", "basis"} <= set(cost["model_only"])
    assert isinstance(cost["estimate_note"], str) and cost["estimate_note"]
    vol = res["volume"]
    assert {"slots_per_frame", "refresh_s", "seconds_per_frame_1core", "cameras_per_core",
            "cameras_per_core_at_70pct", "slots_per_core"} <= set(vol)
    assert set(res["volume_by_scenario"]) == {"comercial", "uvg"}
    assert res["model"]["run_id"] == bench["run_dir"].name
    assert res["model"]["onnx"] == "best.onnx"
    assert res["data"]["split"] == "val"


@pytest.mark.data
def test_cost_uses_pipeline_basis(bench):
    """Costo = tiempo de 1 núcleo por predicción en modo frame, modelo + preprocesamiento."""
    res = load_json(bench["json"])
    costs = load_yaml(paths.configs_dir() / "costs.yaml")
    usd_h, gtq = float(costs["cpu_server_usd_per_hour"]), float(costs["gtq_per_usd"])
    cost = res["cost"]
    assert cost["cpu_server_usd_per_hour"] == usd_h and cost["gtq_per_usd"] == gtq

    slots = res["volume"]["slots_per_frame"]
    assert slots == B.SLOTS_PER_FRAME
    frame_ms = res["cpu_1_thread"]["per_frame"]["mean_ms"]
    pre_ms = res["preprocess_ms_per_patch"]
    pipeline_pred_s = slots / ((frame_ms + slots * pre_ms) / 1000.0)
    assert cost["pipeline_pred_s"] == pytest.approx(pipeline_pred_s, rel=1e-9)
    seconds = 1000.0 / pipeline_pred_s
    assert cost["core_seconds_per_1000"] == pytest.approx(seconds, rel=1e-9)
    assert cost["usd"] == pytest.approx(seconds / 3600 * usd_h, rel=1e-9)
    assert cost["q"] == pytest.approx(seconds / 3600 * usd_h * gtq, rel=1e-9)
    assert res["costo_por_1000_pred_Q"] == pytest.approx(cost["q"], rel=1e-12)

    # Misma base que el volumen: segundos por frame / 46 = segundos por predicción
    frame_s = res["volume"]["seconds_per_frame_1core"]
    assert frame_s == pytest.approx((frame_ms + slots * pre_ms) / 1000.0, rel=1e-9)
    assert cost["core_seconds_per_1000"] == pytest.approx(frame_s / slots * 1000, rel=1e-9)
    assert res["volume"]["cameras_per_core"] == math.floor(res["volume"]["refresh_s"] / frame_s)

    # Variante solo modelo: throughput del frame con 1 hilo, sin preprocesar
    only = cost["model_only"]
    thr = res["cpu_1_thread"]["per_frame"]["throughput_pred_s"]
    assert only["core_seconds_per_1000"] == pytest.approx(1000.0 / thr, rel=1e-9)
    assert only["q"] == pytest.approx(1000.0 / thr / 3600 * usd_h * gtq, rel=1e-9)
    assert res["costo_por_1000_pred_Q"] >= only["q"]

    # Volumen por escenario con el mismo tiempo por frame
    for sc in res["volume_by_scenario"].values():
        assert sc["core_utilization"] == pytest.approx(sc["n_camaras"] * frame_s / sc["refresh_s"], rel=1e-9)


@pytest.mark.data
def test_benchmark_only_reads_val(bench):
    assert bench["rec"].names == {"val"}
    assert all(not allow for _, allow in bench["rec"].calls)


@pytest.mark.data
def test_benchmark_json_has_no_local_paths(bench):
    text = bench["json"].read_text(encoding="utf-8")
    assert_no_local_paths(text, bench["root"], bench["run_dir"])


@pytest.mark.data
def test_mismatched_run_fails_and_writes_nothing(exported, tmp_path, capsys):
    other = tmp_path / "otro_run"
    shutil.copytree(exported["run_dir"], other)
    (other / "benchmark.json").unlink(missing_ok=True)
    out = tmp_path / "out"
    rc = B.main(["--onnx", str(exported["onnx"]), "--run", str(other), "--n", "2", "--n-frame", "1",
                 "--out", str(out), "--no-gpu"])
    assert rc != 0
    assert "no de" in capsys.readouterr().err
    assert not out.exists()
    assert not (other / "benchmark.json").exists()


@pytest.mark.data
def test_onnx_without_card_fails_with_run(exported, tmp_path, capsys):
    lonely = tmp_path / "solo" / "modelo.onnx"
    lonely.parent.mkdir()
    shutil.copy(exported["onnx"], lonely)
    out = tmp_path / "out"
    rc = B.main(["--onnx", str(lonely), "--run", str(exported["run_dir"]), "--n", "2", "--n-frame", "1",
                 "--out", str(out), "--no-gpu"])
    assert rc != 0
    assert "modelo.json" in capsys.readouterr().err
    assert not out.exists()
    assert sorted(p.name for p in lonely.parent.iterdir()) == ["modelo.onnx"]


def test_missing_onnx_fails(tmp_path, capsys):
    out = tmp_path / "out"
    rc = B.main(["--onnx", str(tmp_path / "no_existe.onnx"), "--out", str(out), "--no-gpu"])
    assert rc != 0
    assert "Error" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.data
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere CUDA")
def test_gpu_block_when_cuda(exported, tmp_path):
    run_copy = tmp_path / exported["run_dir"].name
    shutil.copytree(exported["run_dir"], run_copy)
    onnx_dir = tmp_path / "m"
    onnx_dir.mkdir()
    shutil.copy(exported["onnx"], onnx_dir / "best.onnx")
    shutil.copy(exported["onnx"].with_suffix(".json"), onnx_dir / "best.json")
    out = tmp_path / "out"
    rc = B.main(["--onnx", str(onnx_dir / "best.onnx"), "--run", str(run_copy), "--n", "4",
                 "--n-frame", "2", "--out", str(out)])
    assert rc == 0
    gpu = load_json(out / "benchmark.json")["gpu_torch"]
    assert gpu is not None and gpu["engine"] == "torch"
    assert gpu["per_slot"]["batch"] == 1 and gpu["per_frame"]["batch"] == B.SLOTS_PER_FRAME
    assert gpu["per_slot"]["p50_ms"] > 0 and gpu["per_frame"]["p95_ms"] >= gpu["per_frame"]["p50_ms"]
