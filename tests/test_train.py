"""Pruebas de train.py: scheduler, criterio del mejor modelo, config y prueba de humo real.

La prueba de humo (marca ``data``) entrena B2 (mAlexNet) 1 época con el 2 % de train y val
en una carpeta temporal (``--runs-dir``) y verifica que ``runs/<id>/`` quede completo.
"""

from __future__ import annotations

import ast
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from cupo import paths  # noqa: E402
from cupo import train as T  # noqa: E402
from cupo.runs import load_npz, load_yaml  # noqa: E402

from run_helpers import (  # noqa: E402
    CONFIGS, E6_COMMON, E6_HISTORY, E6_NPZ, E6_SUMMARY, HAS_DATA, PYTHON, REPO,
    install_split_recorders, load_strict_json, tiny_torch_cfg,
)

needs_data = pytest.mark.skipif(not HAS_DATA, reason="dataset no disponible en DATA_ROOT")
SMOKE_LIMIT_S = 180.0


# ================================================================ unitarias

def test_lr_factor_constant_and_warmup():
    assert T.lr_factor(0, 100, 0, "none", 0.0) == 1.0
    assert T.lr_factor(99, 100, 0, "constant", 0.0) == 1.0
    # warmup lineal de 4 pasos: 1/4, 2/4, 3/4, 4/4
    assert [T.lr_factor(s, 100, 4, "none", 0.0) for s in range(5)] == [0.25, 0.5, 0.75, 1.0, 1.0]


def test_lr_factor_cosine_by_hand():
    # total 10, sin warmup: progreso s/10 → 0.5·(1 + cos(π·s/10))
    assert T.lr_factor(0, 10, 0, "cosine", 0.0) == pytest.approx(1.0)
    assert T.lr_factor(5, 10, 0, "cosine", 0.0) == pytest.approx(0.5)
    assert T.lr_factor(10, 10, 0, "cosine", 0.0) == pytest.approx(0.0, abs=1e-12)
    assert T.lr_factor(5, 10, 0, "cosine", 0.2) == pytest.approx(0.2 + 0.8 * 0.5)
    # con 2 pasos de warmup, el coseno arranca en el paso 2: progreso (4-2)/(10-2) = 0.25
    assert T.lr_factor(4, 10, 2, "cosine", 0.0) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.25)))


def test_lr_factor_unknown_schedule_raises():
    with pytest.raises(ValueError):
        T.lr_factor(3, 10, 0, "step", 0.0)


def test_scheduler_constant_keeps_lr():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-3)
    sch = T.build_scheduler(opt, {"scheduler": "none", "warmup_epochs": 0}, epochs=2, steps_per_epoch=5)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert lrs == [1e-3] * 10


def test_is_better_value_then_loss():
    assert T.is_better(10.0, 0.5, -np.inf, np.inf)
    assert T.is_better(11.0, 0.9, 10.0, 0.1)          # más valor gana aunque la pérdida sea peor
    assert not T.is_better(9.0, 0.01, 10.0, 0.5)
    assert T.is_better(10.0, 0.4, 10.0, 0.5)          # empate de valor: gana menor val loss
    assert not T.is_better(10.0, 0.6, 10.0, 0.5)
    assert T.is_better(1e6 * (1 + 1e-12), 0.4, 1e6, 0.5)  # empate dentro de la tolerancia


def test_effective_workers():
    assert T.effective_workers(100, 6) == 0
    assert T.effective_workers(T.SMALL_DATASET, 6) == 0
    assert T.effective_workers(T.SMALL_DATASET + 1, 6) == 6


def test_resolve_config_b2_defaults_and_overrides():
    args = T.parse_args(["--config", str(CONFIGS / "b2_malexnet.yaml"), "--seed", "7",
                         "--subset", "0.02", "--epochs", "1", "--n-boot", "20"])
    cfg = T.resolve_config(args.config, T.cli_overrides(args))
    assert cfg["name"] == "b2_malexnet"
    assert cfg["model"] == {"name": "malexnet", "pretrained": False, "freeze": "none", "num_classes": 2}
    assert cfg["optim"]["scheduler"] == "none"       # B2 usa lr constante
    assert cfg["data"]["augment"] is False
    assert cfg["train"]["seed"] == 7 and cfg["train"]["epochs"] == 1
    assert cfg["data"]["subset"] == 0.02 and cfg["eval"]["n_boot"] == 20
    assert cfg["select"]["scenario"] == "comercial"


@pytest.mark.parametrize("name", ["b2_malexnet", "k1_malexnet_aug", "k2_mnv3_head", "k3_mnv3_ft",
                                  "k4_mnv3_ft_cost", "k5_efficientnet_b0"])
def test_torch_configs_resolve_and_build(name):
    from cupo.models import build_model, count_params

    cfg = T.resolve_config(CONFIGS / f"{name}.yaml")
    assert cfg["name"] == name
    assert cfg["select"]["scenario"] in ("comercial", "uvg")
    m = cfg["model"]
    model = build_model(m["name"], pretrained=False, freeze=m["freeze"],
                        input_size=int(cfg["data"]["input_size"]))
    total, trainable = count_params(model)
    assert (trainable < total) == (m["freeze"] != "none")


def test_config_expectations_per_iteration():
    k = {n: T.resolve_config(CONFIGS / f"{n}.yaml") for n in
         ("k1_malexnet_aug", "k2_mnv3_head", "k3_mnv3_ft", "k4_mnv3_ft_cost", "k5_efficientnet_b0")}
    assert k["k1_malexnet_aug"]["data"]["augment"] and k["k1_malexnet_aug"]["optim"]["scheduler"] == "cosine"
    assert k["k2_mnv3_head"]["model"]["freeze"] == "backbone" and k["k2_mnv3_head"]["model"]["pretrained"]
    assert k["k3_mnv3_ft"]["model"]["freeze"] == "none"
    assert k["k4_mnv3_ft_cost"]["loss"]["cost_weight_busy"] > 1
    assert k["k5_efficientnet_b0"]["model"]["name"] == "efficientnet_b0"


def test_history_columns_include_e6_plus_extras():
    assert T.HISTORY_COLUMNS[:len(E6_HISTORY)] == E6_HISTORY
    assert {"lr_start", "train_nll"} <= set(T.HISTORY_COLUMNS)


def test_train_cost_by_hand():
    usd, q = T.train_cost(3600.0)
    assert usd == pytest.approx(0.35) and q == pytest.approx(0.35 * 7.7)
    usd_cpu, _ = T.train_cost(7200.0, "cpu_server_usd_per_hour")
    assert usd_cpu == pytest.approx(0.10)


def test_nll_matches_hand_value():
    logits = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    y = np.array([0, 1])
    assert T.nll(logits, y) == pytest.approx((math.log1p(math.exp(-2)) + math.log(2)) / 2, rel=1e-6)


def test_train_source_never_mentions_protected_splits():
    """Análisis estático: train.py y train_classical.py no piden test/ood ni usan allow_test."""
    for rel in ("src/cupo/train.py", "scripts/train_classical.py", "src/cupo/calibrate.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    assert kw.arg != "allow_test", f"{rel} usa allow_test"
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                if name == "load_split":
                    consts = {a.value for a in node.args if isinstance(a, ast.Constant)}
                    assert consts <= {"train", "val"}, f"{rel}: {ast.dump(node)}"
        assert "CNRPark" not in src


# ===================================================== invariante en ejecución

@pytest.mark.data
@needs_data
def test_run_training_only_requests_train_and_val(monkeypatch, tmp_path):
    rec, inner = install_split_recorders(monkeypatch)
    cfg = tiny_torch_cfg(subset=0.005)
    run_dir = T.run_training(cfg, run_name="inv", runs_root=tmp_path)
    assert rec.calls and rec.names == {"train", "val"}
    assert all(allow is False for _, allow in rec.calls)
    assert set(inner.names) <= {"train", "val"}
    # las predicciones de val solo contienen fechas de val
    from cupo.data import split_dates

    preds = load_npz(run_dir / "val_preds.npz")
    assert set(preds["date"].tolist()) <= split_dates("val")
    assert Path(run_dir).parent == tmp_path.resolve()


# ============================================================ prueba de humo

def _smoke(runs_dir: Path, run_name: str) -> tuple[Path, float, subprocess.CompletedProcess]:
    cmd = [PYTHON, "-m", "cupo.train", "--config", str(CONFIGS / "b2_malexnet.yaml"),
           "--subset", "0.02", "--epochs", "1", "--runs-dir", str(runs_dir), "--run-name", run_name]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=600)
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, proc.stderr[-3000:]
    return runs_dir / run_name, elapsed, proc


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    if not HAS_DATA:
        pytest.skip("dataset no disponible en DATA_ROOT")
    root = tmp_path_factory.mktemp("runs_smoke")
    a = _smoke(root, "b2_smoke_a")
    b = _smoke(root, "b2_smoke_b")
    return {"a": a, "b": b, "root": root}


@pytest.fixture(scope="module")
def smoke_dir(smoke_runs) -> Path:
    return smoke_runs["a"][0]


@pytest.mark.data
@needs_data
def test_smoke_finishes_under_three_minutes(smoke_runs):
    for key in ("a", "b"):
        _, elapsed, _ = smoke_runs[key]
        assert elapsed < SMOKE_LIMIT_S, f"humo {key}: {elapsed:.1f} s"


@pytest.mark.data
@needs_data
def test_smoke_writes_only_in_runs_dir(smoke_runs):
    root = smoke_runs["root"]
    assert sorted(p.name for p in root.iterdir()) == ["b2_smoke_a", "b2_smoke_b"]


@pytest.mark.data
@needs_data
def test_smoke_generates_all_e6_files(smoke_dir):
    files = {p.name for p in smoke_dir.iterdir()}
    missing = set(E6_COMMON + ["best.pt"]) - files
    assert not missing, f"faltan {missing}"
    assert not any(n.startswith(("metrics_test", "metrics_ood", "test_", "ood_")) for n in files)
    assert (smoke_dir / "curves.png").stat().st_size > 1000


@pytest.mark.data
@needs_data
def test_smoke_config_has_effective_seed_and_subset(smoke_dir):
    cfg = load_yaml(smoke_dir / "config.yaml")
    assert cfg["train"]["seed"] == 42
    assert cfg["data"]["subset"] == 0.02
    assert cfg["train"]["epochs"] == 1
    assert cfg["name"] == "b2_malexnet"


@pytest.mark.data
@needs_data
def test_smoke_env_json(smoke_dir):
    env = load_strict_json(smoke_dir / "env.json")
    for key in ("git_commit", "python", "torch", "torchvision", "gpu", "cpu", "date"):
        assert key in env
    assert env["python"].startswith("3.")


@pytest.mark.data
@needs_data
def test_smoke_history_columns_and_constant_lr(smoke_dir):
    hist = pd.read_csv(smoke_dir / "history.csv")
    assert list(hist.columns)[:len(E6_HISTORY)] == E6_HISTORY
    assert {"lr_start", "train_nll"} <= set(hist.columns)
    assert len(hist) == 1 and hist["epoch"].tolist() == [1]
    row = hist.iloc[0]
    assert row["lr"] == pytest.approx(1e-3) and row["lr_start"] == pytest.approx(1e-3)  # B2: constante
    assert np.isfinite(row[["train_loss", "train_nll", "val_loss", "val_value"]].astype(float)).all()
    # sin pesos ni smoothing, la pérdida de optimización y la NLL coinciden
    assert row["train_nll"] == pytest.approx(row["train_loss"], rel=1e-3)
    assert 0 <= row["train_acc"] <= 1 and 0 <= row["val_acc"] <= 1


@pytest.mark.data
@needs_data
def test_smoke_summary_keys_and_values(smoke_dir):
    s = load_strict_json(smoke_dir / "summary.json")
    missing = set(E6_SUMMARY) - set(s)
    assert not missing, f"faltan {missing}"
    assert s["run_id"] == "b2_smoke_a" and s["name"] == "b2_malexnet" and s["model"] == "malexnet"
    assert s["params_total"] == s["params_trainable"] == 42_460
    assert s["best_epoch"] == 1
    assert s["train_cost_usd"] == pytest.approx(s["train_time_s"] / 3600 * 0.35, rel=1e-9)
    assert s["train_cost_q"] == pytest.approx(s["train_cost_usd"] * 7.7, rel=1e-9)
    hist = pd.read_csv(smoke_dir / "history.csv")
    value = load_strict_json(smoke_dir / "value_val.json")
    assert s["val_value_comercial"] == pytest.approx(value["comercial"]["objective_value"])
    assert s["val_value_uvg"] == pytest.approx(value["uvg"]["objective_value"])
    # el valor del run con sus umbrales coincide con el val_value de la mejor época
    assert s["val_value_comercial"] == pytest.approx(hist.loc[0, "val_value"], rel=1e-9)


@pytest.mark.data
@needs_data
def test_smoke_thresholds_ordered(smoke_dir):
    thr = load_strict_json(smoke_dir / "thresholds.json")
    assert list(thr) == ["comercial", "uvg"]  # el escenario de selección primero
    for v in thr.values():
        assert 0.0 <= v["t_L"] <= v["t_H"] <= 1.0
        assert v["t_L"] <= 0.5 <= v["t_H"]


@pytest.mark.data
@needs_data
def test_smoke_value_val_structure(smoke_dir):
    value = load_strict_json(smoke_dir / "value_val.json")
    thr = load_strict_json(smoke_dir / "thresholds.json")
    for scen in ("comercial", "uvg"):
        v = value[scen]
        for key in ("t_L", "t_H", "objective_value", "components", "ci95"):
            assert key in v
        lo, hi = v["ci95"]["objective"]
        assert lo <= hi
        assert (v["t_L"], v["t_H"]) == (thr[scen]["t_L"], thr[scen]["t_H"])


@pytest.mark.data
@needs_data
def test_smoke_val_preds_and_metrics(smoke_dir):
    from cupo.data import load_split, split_dates, subsample

    preds = load_npz(smoke_dir / "val_preds.npz")
    assert set(E6_NPZ) <= set(preds)
    n = preds["y"].size
    expected = subsample(load_split("val"), 0.02, 42)
    assert n == len(expected)
    assert preds["path"].tolist() == expected["path"].tolist()
    assert ((preds["p"] >= 0) & (preds["p"] <= 1)).all()
    assert set(preds["date"].tolist()) <= split_dates("val")
    metrics = load_strict_json(smoke_dir / "metrics_val.json")
    for key in ("binary", "ci95", "three_state", "thresholds", "breakdown"):
        assert key in metrics
    assert metrics["n"] == n
    b = metrics["binary"]
    assert b["accuracy"] == pytest.approx(np.mean((preds["p"] >= 0.5) == (preds["y"] == 1)))


@pytest.mark.data
@needs_data
def test_smoke_best_checkpoint(smoke_dir):
    ckpt = torch.load(smoke_dir / "best.pt", map_location="cpu", weights_only=True)
    assert set(ckpt) == {"state_dict", "model_cfg", "input_size", "epoch"}
    assert ckpt["input_size"] == 224 and ckpt["epoch"] == 1
    assert ckpt["model_cfg"]["name"] == "malexnet"


@pytest.mark.data
@needs_data
def test_smoke_all_json_are_strict(smoke_dir):
    for f in smoke_dir.glob("*.json"):
        load_strict_json(f)


@pytest.mark.data
@needs_data
def test_same_seed_gives_same_train_loss(smoke_runs):
    ha = pd.read_csv(smoke_runs["a"][0] / "history.csv")
    hb = pd.read_csv(smoke_runs["b"][0] / "history.csv")
    assert ha.loc[0, "train_loss"] == pytest.approx(hb.loc[0, "train_loss"], rel=1e-6)
    assert ha.loc[0, "train_acc"] == hb.loc[0, "train_acc"]
    assert ha.loc[0, "val_loss"] == pytest.approx(hb.loc[0, "val_loss"], rel=1e-5)
    pa = load_npz(smoke_runs["a"][0] / "val_preds.npz")["p"]
    pb = load_npz(smoke_runs["b"][0] / "val_preds.npz")["p"]
    np.testing.assert_allclose(pa, pb, atol=1e-4)


# ============================================ carpeta del run y subset efectivo

@pytest.fixture
def captured_training(tmp_path, monkeypatch):
    """Reemplaza run_training por un registro y apunta runs/ a tmp_path."""
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "runs_dir", lambda: root)
    calls = []

    def fake_run_training(cfg, run_name=None, runs_root=None):
        calls.append({"cfg": cfg, "run_name": run_name, "runs_root": runs_root})
        return Path(runs_root) / (run_name or "x")

    monkeypatch.setattr(T, "run_training", fake_run_training)
    return root, calls


def _yaml_with_subset(tmp_path: Path, subset: float) -> Path:
    from cupo.runs import save_yaml

    cfg = load_yaml(CONFIGS / "b2_malexnet.yaml")
    cfg["data"]["subset"] = subset
    out = tmp_path / f"b2_subset_{subset}.yaml"
    save_yaml(out, cfg)
    return out


def test_main_cli_subset_goes_to_smoke(captured_training):
    root, calls = captured_training
    assert T.main(["--config", str(CONFIGS / "b2_malexnet.yaml"), "--subset", "0.02", "--epochs", "1"]) == 0
    assert calls[0]["cfg"]["data"]["subset"] == 0.02
    assert calls[0]["runs_root"] == root / "_smoke"


def test_main_full_subset_goes_to_runs(captured_training):
    root, calls = captured_training
    assert T.main(["--config", str(CONFIGS / "b2_malexnet.yaml")]) == 0
    assert calls[0]["cfg"]["data"]["subset"] == 1.0
    assert calls[0]["runs_root"] == root


def test_main_yaml_subset_goes_to_smoke(captured_training, tmp_path):
    root, calls = captured_training
    assert T.main(["--config", str(_yaml_with_subset(tmp_path, 0.1))]) == 0
    assert calls[0]["cfg"]["data"]["subset"] == 0.1
    assert calls[0]["runs_root"] == root / "_smoke"


def test_main_cli_subset_overrides_yaml(captured_training, tmp_path):
    root, calls = captured_training
    assert T.main(["--config", str(_yaml_with_subset(tmp_path, 0.1)), "--subset", "1.0"]) == 0
    assert calls[0]["cfg"]["data"]["subset"] == 1.0
    assert calls[0]["runs_root"] == root


def test_main_explicit_runs_dir_is_respected(captured_training, tmp_path):
    _, calls = captured_training
    explicit = tmp_path / "mis_runs"
    T.main(["--config", str(CONFIGS / "b2_malexnet.yaml"), "--subset", "0.02", "--runs-dir", str(explicit)])
    T.main(["--config", str(CONFIGS / "b2_malexnet.yaml"), "--runs-dir", str(explicit)])
    assert [c["runs_root"] for c in calls] == [explicit, explicit]


@pytest.mark.data
@needs_data
def test_run_training_in_smoke_root_writes_effective_subset(monkeypatch, tmp_path):
    """Run real diminuto en la carpeta elegida por choose_runs_root (runs/ falso)."""
    from cupo.runs import choose_runs_root

    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path / "runs")
    install_split_recorders(monkeypatch)
    cfg = tiny_torch_cfg(subset=0.005)
    root = choose_runs_root(None, cfg["data"]["subset"])
    run_dir = T.run_training(cfg, run_name="smoke_root", runs_root=root)
    assert Path(run_dir) == (tmp_path / "runs" / "_smoke" / "smoke_root").resolve()
    assert load_yaml(run_dir / "config.yaml")["data"]["subset"] == 0.005
    assert load_strict_json(run_dir / "summary.json")["subset"] == 0.005


# ===================================================================== K6

def test_k6_config_resolves_and_builds():
    from cupo.models import build_model, count_params

    k6 = T.resolve_config(CONFIGS / "k6_mnv3_ft_seeds.yaml")
    k3 = T.resolve_config(CONFIGS / "k3_mnv3_ft.yaml")
    assert k6["name"] == "k6_mnv3_ft_seeds"
    for section in ("model", "data", "optim", "loss", "train", "select", "eval"):
        assert k6[section] == k3[section], section
    m = k6["model"]
    model = build_model(m["name"], pretrained=False, freeze=m["freeze"],
                        input_size=int(k6["data"]["input_size"]))
    total, trainable = count_params(model)
    assert total == trainable > 0
    with torch.no_grad():
        assert model.eval()(torch.randn(2, 3, 224, 224)).shape == (2, 2)


def test_k6_seed_override():
    args = T.parse_args(["--config", str(CONFIGS / "k6_mnv3_ft_seeds.yaml"), "--seed", "43"])
    cfg = T.resolve_config(args.config, T.cli_overrides(args))
    assert cfg["train"]["seed"] == 43
    assert cfg["data"]["subset"] == 1.0
