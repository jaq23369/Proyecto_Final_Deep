"""Pruebas de scripts/gradcam.py: Grad-CAM a mano, capa objetivo, selección de ejemplos y CLI."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from cupo import paths  # noqa: E402
from cupo.runs import load_npz, load_run, save_npz  # noqa: E402

from export_helpers import fake_classical_run, load_gradcam, torch_run_copy  # noqa: E402
from run_helpers import HAS_DATA, SplitRecorder  # noqa: E402

G = load_gradcam()
CATEGORIES = {G.CAT_BUSY_AS_FREE, G.CAT_FREE_AS_BUSY, G.CAT_HIT_BUSY, G.CAT_HIT_FREE, G.CAT_VERIFY}


# ---------------------------------------------------------------------------
# Grad-CAM calculado a mano
# ---------------------------------------------------------------------------

def _toy_model():
    """conv 1x1 identidad → ReLU → promedio global → lineal; la capa objetivo es la ReLU."""
    conv = nn.Conv2d(2, 2, kernel_size=1, bias=False)
    lin = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.eye(2).view(2, 2, 1, 1))
        lin.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, -1.0]]))
    return nn.Sequential(conv, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), lin).eval()


def test_cam_by_hand_on_toy_model():
    model = _toy_model()
    # A = ReLU(x): canal 0 [[1,2],[3,4]], canal 1 [[4,0],[0,0]]
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[4.0, 0.0], [0.0, -1.0]]]])
    x = x.repeat(2, 1, 1, 1)
    cam = G.GradCAM(model, model[1])
    maps, logits = cam(x, np.array([1, 0]))
    cam.close()
    # z0 = 0·2.5 + 1·1 = 1;  z1 = 1·2.5 - 1·1 = 1.5
    np.testing.assert_allclose(logits[0], [1.0, 1.5], atol=1e-6)
    # Ocupado: d(z1 - z0)/dA_k = (W1k - W0k)/4 → w = [0.25, -0.5]
    # ReLU(0.25·A0 - 0.5·A1) = [[0, 0.5], [0.75, 1]] (ya en [0, 1])
    np.testing.assert_allclose(maps[0], [[0.0, 0.5], [0.75, 1.0]], atol=1e-6)
    # Libre: w = [-0.25, 0.5] → ReLU = [[1.75, 0], [0, 0]] → normalizado [[1, 0], [0, 0]]
    np.testing.assert_allclose(maps[1], [[1.0, 0.0], [0.0, 0.0]], atol=1e-6)


def test_cam_constant_map_is_zero_not_nan():
    model = _toy_model()
    x = -torch.ones(1, 2, 2, 2)  # A = 0 → mapa nulo
    cam = G.GradCAM(model, model[1])
    maps, _ = cam(x, np.array([1]))
    cam.close()
    assert np.isfinite(maps).all() and np.allclose(maps, 0.0)


def test_hook_is_removed_on_close():
    model = _toy_model()
    cam = G.GradCAM(model, model[1])
    assert len(model[1]._forward_hooks) == 1
    cam.close()
    assert len(model[1]._forward_hooks) == 0


def test_malexnet_target_is_relu_after_conv3():
    from cupo.models import build_model

    model = build_model("malexnet", num_classes=2, pretrained=False, freeze="none", input_size=224)
    layer = G.target_layer(model, "malexnet")
    assert isinstance(layer, nn.ReLU)
    block = model.features[-1]
    conv = block[0]
    assert isinstance(conv, nn.Conv2d) and conv.in_channels == 20 and conv.out_channels == 30
    assert block[1] is layer
    # Es la última capa conv de la red (después solo hay MaxPool y el clasificador)
    assert not any(isinstance(m, nn.Conv2d) for m in list(block)[2:])
    assert not any(isinstance(m, nn.Conv2d) for m in model.classifier.modules())


def test_mobilenet_target_is_last_features_block():
    from cupo.models import build_model

    model = build_model("mobilenet_v3_small", num_classes=2, pretrained=False, freeze="none", input_size=224)
    layer = G.target_layer(model, "mobilenet_v3_small")
    assert layer is model.features[-1]
    assert any(isinstance(m, nn.Conv2d) for m in layer.modules())
    with pytest.raises(ValueError):
        G.target_layer(model, "vgg16")


def test_cam_matches_manual_on_malexnet():
    """Grad-CAM de mAlexNet = cálculo manual con la activación post-ReLU de conv3."""
    from cupo.models import build_model

    torch.manual_seed(0)
    model = build_model("malexnet", num_classes=2, pretrained=False, freeze="none", input_size=224).eval()
    G.disable_inplace(model)
    x = torch.randn(3, 3, 224, 224)
    target = np.array([1, 0, 1])

    cam = G.GradCAM(model, G.target_layer(model, "malexnet"))
    maps, logits = cam(x, target)
    cam.close()

    # Manual: features[0], features[1], conv3 + ReLU → A; luego MaxPool + clasificador
    h = model.features[1](model.features[0](x))
    a = torch.relu(model.features[2][0](h))
    a.retain_grad()
    z = model.classifier(model.features[2][2](a))
    sign = torch.tensor([1.0, -1.0, 1.0])
    (sign * (z[:, 1] - z[:, 0])).sum().backward()
    w = a.grad.mean(dim=(2, 3), keepdim=True)
    m = torch.relu((w * a.detach()).sum(1, keepdim=True))
    m = F.interpolate(m, size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
    flat = m.flatten(1)
    lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
    expected = ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(m).detach().numpy()

    np.testing.assert_allclose(logits, z.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(maps, expected, atol=1e-5)
    assert maps.shape == (3, 224, 224)
    assert maps.min() >= 0.0 and maps.max() <= 1.0 + 1e-6


def test_disable_inplace_turns_off_all_inplace():
    from cupo.models import build_model

    model = build_model("malexnet", num_classes=2, pretrained=False, freeze="none", input_size=224)
    assert any(getattr(m, "inplace", False) is True for m in model.modules())
    G.disable_inplace(model)
    assert not any(getattr(m, "inplace", False) is True for m in model.modules())


# ---------------------------------------------------------------------------
# Selección de ejemplos
# ---------------------------------------------------------------------------

def _df(rows):
    return pd.DataFrame(rows, columns=["y", "p"]).assign(path="x.jpg", weather="SUNNY", camera=1)


def test_categorize_and_decision_by_hand():
    df = _df([(1, 0.1), (0, 0.8), (1, 0.9), (0, 0.2), (1, 0.45), (0, 0.55), (1, 0.3), (0, 0.7)])
    cat = G.categorize(df, t_L=0.3, t_H=0.7).tolist()
    assert cat == [G.CAT_BUSY_AS_FREE, G.CAT_FREE_AS_BUSY, G.CAT_HIT_BUSY, G.CAT_HIT_FREE,
                   G.CAT_VERIFY, G.CAT_VERIFY, G.CAT_VERIFY, G.CAT_FREE_AS_BUSY]
    # p = t_L cae en verificar; p = t_H es ocupado
    dec = G.decision_class(df.p.to_numpy(), 0.3, 0.7).tolist()
    assert dec == [0, 1, 1, 0, 0, 1, 0, 1]


def test_select_examples_errors_first_busy_as_free_most_confident():
    rows = ([(1, p) for p in (0.05, 0.2, 0.1)]            # ocupado→libre
            + [(0, p) for p in (0.9, 0.99)]               # libre→ocupado
            + [(1, 0.5), (0, 0.5), (1, 0.6)]              # por verificar
            + [(1, 0.95)] * 6 + [(0, 0.02)] * 6)          # aciertos
    df = _df(rows)
    sel = G.select_examples(df, n=8, t_L=0.3, t_H=0.7)
    assert len(sel) == 8
    cats = sel.category.tolist()
    # ceil(8/2) = 4 errores: los 3 ocupado→libre (p ascendente) y el libre→ocupado más alto
    assert cats[:4] == [G.CAT_BUSY_AS_FREE] * 3 + [G.CAT_FREE_AS_BUSY]
    assert sel.p.tolist()[:4] == [0.05, 0.1, 0.2, 0.99]
    assert cats.count(G.CAT_VERIFY) == 2
    assert cats.count(G.CAT_HIT_BUSY) == 1 and cats.count(G.CAT_HIT_FREE) == 1


def test_select_examples_is_deterministic():
    rng = np.random.default_rng(0)
    df = _df(list(zip(rng.integers(0, 2, 200), rng.random(200))))
    a = G.select_examples(df, 10, 0.3, 0.7)
    b = G.select_examples(df, 10, 0.3, 0.7)
    pd.testing.assert_frame_equal(a, b)


def test_empty_categories_are_filled_with_warning(capsys):
    # Sin errores ni "por verificar"; solo 1 acierto libre y 10 aciertos ocupados
    df = _df([(1, 0.95)] * 10 + [(0, 0.05)])
    df["path"] = [f"img_{i}.jpg" for i in range(len(df))]
    sel = G.select_examples(df, n=8, t_L=0.3, t_H=0.7)
    out = capsys.readouterr().out
    assert len(sel) == 8
    assert sel.path.is_unique
    assert "Aviso" in out
    for c in (G.CAT_BUSY_AS_FREE, G.CAT_FREE_AS_BUSY, G.CAT_VERIFY):
        assert c in out
    assert "se completaron 3" in out
    assert set(sel.category) <= CATEGORIES
    assert (sel.category == G.CAT_HIT_FREE).sum() == 1


def test_n_larger_than_available_returns_all(capsys):
    df = _df([(1, 0.95), (0, 0.05), (1, 0.1)])
    sel = G.select_examples(df, n=10, t_L=0.3, t_H=0.7)
    assert len(sel) == 3
    assert "Aviso" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

def test_data_relative_for_foreign_absolute_path():
    foreign = r"D:\otra_maquina\Data\CNR-EXT\PATCHES\SUNNY\2015-11-12\camera1\S_x.jpg"
    assert G.data_relative(foreign) == "CNR-EXT/PATCHES/SUNNY/2015-11-12/camera1/S_x.jpg"
    rel = G.resolve_patch_path(foreign)
    assert rel == paths.data_root() / "CNR-EXT" / "PATCHES" / "SUNNY" / "2015-11-12" / "camera1" / "S_x.jpg"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["test", "ood"])
def test_protected_splits_are_rejected(split, tmp_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no debe cargar el run")

    monkeypatch.setattr(G, "load_run", boom)
    with pytest.raises(SystemExit) as exc:
        G.main(["--run", "x", "--split", split, "--n", "2", "--out", str(tmp_path / "fig")])
    assert exc.value.code != 0
    assert "bloqueado" in capsys.readouterr().err
    assert not (tmp_path / "fig").exists()


def test_n_must_be_positive(tmp_path):
    with pytest.raises(SystemExit) as exc:
        G.parse_args(["--run", "x", "--n", "0", "--out", str(tmp_path)])
    assert exc.value.code != 0


def test_default_out_is_results_figures():
    assert Path(G.parse_args(["--run", "x"]).out) == paths.results_dir() / "figures"


def test_classical_run_is_rejected(tmp_path, capsys):
    run_dir = fake_classical_run(tmp_path / "b1_hog_lr_s42_x")
    out = tmp_path / "fig"
    rc = G.main(["--run", str(run_dir), "--n", "2", "--out", str(out), "--device", "cpu"])
    assert rc != 0
    assert "clásico" in capsys.readouterr().err
    assert not out.exists()


@pytest.fixture(scope="module")
def runs_tmp(tmp_path_factory):
    if not HAS_DATA:
        pytest.skip("requiere el dataset en DATA_ROOT")
    root = tmp_path_factory.mktemp("gradcam_runs")
    return {k: torch_run_copy(k, root) for k in ("malexnet", "mnv3_calibrated")}


@pytest.mark.data
def test_calibrated_run_without_logits_fails(runs_tmp, tmp_path, capsys):
    src = runs_tmp["mnv3_calibrated"]
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    preds = load_npz(dst / "val_preds.npz")
    preds.pop("logits", None)
    save_npz(dst / "val_preds.npz", **preds)
    out = tmp_path / "fig"
    rc = G.main(["--run", str(dst), "--n", "2", "--out", str(out), "--device", "cpu"])
    assert rc != 0
    assert "logits" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.data
def test_uncalibrated_run_without_logits_uses_p(runs_tmp, tmp_path):
    src = runs_tmp["malexnet"]
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    preds = load_npz(dst / "val_preds.npz")
    preds.pop("logits", None)
    save_npz(dst / "val_preds.npz", **preds)
    table = G.load_val_table(load_run(dst, device="cpu"))
    np.testing.assert_allclose(table.p.to_numpy(), preds["p"])


@pytest.mark.data
def test_calibrated_table_applies_temperature(runs_tmp):
    from cupo.evaluate import probs_from_logits

    run = load_run(runs_tmp["mnv3_calibrated"], device="cpu")
    assert run.calibrated and run.temperature != 1.0
    preds = load_npz(run.run_dir / "val_preds.npz")
    table = G.load_val_table(run)
    np.testing.assert_allclose(table.p.to_numpy(), probs_from_logits(preds["logits"], run.temperature))


def _run_cli(run_dir: Path, out: Path, n: int, device: str = "cpu"):
    import cupo.data

    with pytest.MonkeyPatch.context() as mp:
        rec = SplitRecorder(cupo.data.load_split)
        mp.setattr(cupo.data, "load_split", rec)
        rc = G.main(["--run", str(run_dir), "--split", "val", "--n", str(n), "--out", str(out),
                     "--device", device])
    return rc, rec


@pytest.mark.data
@pytest.mark.parametrize("kind", ["malexnet", "mnv3_calibrated"])
def test_cli_small_n_writes_png_and_csv(runs_tmp, tmp_path, kind):
    run_dir = runs_tmp[kind]
    out = tmp_path / "fig"
    n = 6
    before = sorted(p.name for p in run_dir.iterdir())
    rc, rec = _run_cli(run_dir, out, n)
    assert rc == 0
    stem = f"gradcam_{run_dir.name}_val"
    assert sorted(p.name for p in out.iterdir()) == [f"{stem}.csv", f"{stem}.png"]
    assert (out / f"{stem}.png").stat().st_size > 0
    assert sorted(p.name for p in run_dir.iterdir()) == before  # no escribe en el run
    assert rec.calls == []  # no lee particiones: usa val_preds.npz

    csv = pd.read_csv(out / f"{stem}.csv")
    assert len(csv) == n
    assert {"path", "y", "p", "weather", "camera", "category", "explained", "p_recomputed"} <= set(csv.columns)
    assert set(csv.category) <= CATEGORIES
    run = load_run(run_dir, device="cpu")
    scenario, t_L, t_H = G.operating_thresholds(run)
    assert csv.category.tolist() == G.categorize(csv, t_L, t_H).tolist()
    assert csv.explained.tolist() == G.decision_class(csv.p.to_numpy(), t_L, t_H).tolist()
    # p recalculada en fp32 ≈ p del npz (AMP en GPU)
    assert (csv.p_recomputed - csv.p).abs().max() < 0.02
    # Rutas relativas a DATA_ROOT, sin datos locales
    preds_paths = {G.data_relative(p) for p in load_npz(run_dir / "val_preds.npz")["path"]}
    for p in csv.path:
        assert p.startswith("CNR-EXT/PATCHES/"), p
        assert ":" not in p and "\\" not in p and not p.startswith("/")
        assert (paths.data_root() / p).is_file()
        assert p in preds_paths
    text = (out / f"{stem}.csv").read_text(encoding="utf-8")
    assert "Users" not in text and str(paths.data_root()) not in text


@pytest.mark.data
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere CUDA")
def test_cli_on_cuda(runs_tmp, tmp_path):
    rc, _ = _run_cli(runs_tmp["malexnet"], tmp_path / "fig", 4, device="cuda")
    assert rc == 0
    csv = pd.read_csv(next((tmp_path / "fig").glob("*.csv")))
    assert len(csv) == 4 and (csv.p_recomputed - csv.p).abs().max() < 0.02


@pytest.mark.data
def test_script_entrypoint_runs(runs_tmp, tmp_path):
    """``python scripts/gradcam.py`` como lo corre el usuario (proceso aparte, cwd temporal)."""
    import os
    import subprocess

    from run_helpers import PYTHON, REPO

    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    out = tmp_path / "fig"
    proc = subprocess.run([PYTHON, str(REPO / "scripts" / "gradcam.py"), "--run", str(runs_tmp["malexnet"]),
                           "--n", "3", "--out", str(out), "--device", "cpu"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert len(list(out.glob("gradcam_*_val.png"))) == 1
    proc = subprocess.run([PYTHON, str(REPO / "scripts" / "gradcam.py"), "--run", str(runs_tmp["malexnet"]),
                           "--split", "test", "--out", str(out / "x")],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0 and not (out / "x").exists()
