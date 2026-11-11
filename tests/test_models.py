"""Pruebas de models.py: mAlexNet, modelos de torchvision, congelamiento y LRN en fp32."""

from __future__ import annotations

import math
import os

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from cupo import models as Mo  # noqa: E402
from cupo.models import (  # noqa: E402
    FP32LocalResponseNorm, MAlexNet, apply_freeze, backbone_blocks, build_model, count_params,
    head_module, model_size_mb, set_train_mode,
)

HAS_CUDA = torch.cuda.is_available()


def _params(module: nn.Module) -> set[int]:
    return {id(p) for p in module.parameters()}


def _trainable(model: nn.Module) -> set[int]:
    return {id(p) for p in model.parameters() if p.requires_grad}


# ------------------------------------------------------------------ mAlexNet

def test_malexnet_param_count_224():
    """Calculado a mano para 224:
    conv1 3·16·11·11+16 = 5,824; conv2 16·20·5·5+20 = 8,020; conv3 20·30·3·3+30 = 5,430;
    flatten 30·4·4 = 480 → fc1 480·48+48 = 23,088; fc2 48·2+2 = 98. Total 42,460.
    """
    model = build_model("malexnet", pretrained=False)
    assert count_params(model) == (42_460, 42_460)
    assert model.classifier[1].in_features == 480


def test_malexnet_param_count_128():
    """Con 128: 30→15→11→5→3→1 por lado → flatten 30 → fc1 30·48+48 = 1,488. Total 20,860."""
    model = build_model("malexnet", pretrained=False, input_size=128)
    assert model.classifier[1].in_features == 30
    assert count_params(model) == (20_860, 20_860)


@pytest.mark.parametrize("size", [224, 128])
@pytest.mark.parametrize("batch", [1, 3])
def test_malexnet_output_shape(size, batch):
    model = build_model("malexnet", pretrained=False, input_size=size).eval()
    with torch.no_grad():
        out = model(torch.randn(batch, 3, size, size))
    assert out.shape == (batch, 2)
    assert torch.isfinite(out).all()


def test_malexnet_architecture_layers():
    model = MAlexNet()
    convs = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
    assert [(c.out_channels, c.kernel_size, c.stride) for c in convs] == [
        (16, (11, 11), (4, 4)), (20, (5, 5), (1, 1)), (30, (3, 3), (1, 1))]
    lrns = [m for m in model.features.modules() if isinstance(m, FP32LocalResponseNorm)]
    assert len(lrns) == 2 and all(m.lrn.size == 5 for m in lrns)
    pools = [m for m in model.features.modules() if isinstance(m, nn.MaxPool2d)]
    assert len(pools) == 3 and all(p.kernel_size == 3 and p.stride == 2 for p in pools)
    linears = [m for m in model.classifier if isinstance(m, nn.Linear)]
    assert [l.out_features for l in linears] == [48, 2]


def test_malexnet_ignores_pretrained_flag():
    torch.manual_seed(0)
    a = build_model("malexnet", pretrained=True)
    assert count_params(a) == (42_460, 42_460)


def test_model_size_mb_malexnet():
    model = build_model("malexnet", pretrained=False)
    size = model_size_mb(model)
    # 42,460 float32 = 0.162 MB + cabecera del state_dict
    assert 42_460 * 4 / 1024**2 <= size < 0.25


# ------------------------------------------------------ modelos de torchvision

@pytest.mark.parametrize("name", Mo.MODEL_NAMES)
def test_every_model_builds_with_two_classes(name):
    model = build_model(name, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)
    total, trainable = count_params(model)
    assert total == trainable > 0


@pytest.mark.parametrize("name", Mo.TORCHVISION_MODELS)
def test_head_is_replaced_with_two_outputs(name):
    model = build_model(name, pretrained=False)
    head = head_module(model, name)
    last = head if isinstance(head, nn.Linear) else head[-1]
    assert isinstance(last, nn.Linear) and last.out_features == 2


def test_model_names_match_spec():
    assert set(Mo.MODEL_NAMES) == {"malexnet", "mobilenet_v3_small", "mobilenet_v3_large",
                                   "efficientnet_b0", "resnet18", "resnet50"}


def test_unknown_model_and_freeze_raise():
    with pytest.raises(ValueError):
        build_model("vgg16", pretrained=False)
    with pytest.raises(ValueError):
        build_model("malexnet", freeze="todo")


def _weights_cached(filename_prefix: str) -> bool:
    folder = os.path.join(torch.hub.get_dir(), "checkpoints")
    return os.path.isdir(folder) and any(f.startswith(filename_prefix) for f in os.listdir(folder))


@pytest.mark.skipif(
    not (_weights_cached("mobilenet_v3_small") or os.environ.get("CUPO_RUN_SLOW") == "1"),
    reason="prueba lenta: requiere pesos de ImageNet en caché o CUPO_RUN_SLOW=1")
def test_pretrained_weights_load_mobilenet_v3_small():
    """Prueba lenta: los pesos de ImageNet se cargan y cambian respecto a la inicialización."""
    torch.manual_seed(0)
    pre = build_model("mobilenet_v3_small", pretrained=True)
    torch.manual_seed(0)
    rand = build_model("mobilenet_v3_small", pretrained=False)
    w_pre = pre.features[0][0].weight
    w_rand = rand.features[0][0].weight
    assert not torch.allclose(w_pre, w_rand)
    # el BN del stem tiene estadísticas aprendidas (no las iniciales 0/1)
    bn = pre.features[0][1]
    assert not torch.allclose(bn.running_mean, torch.zeros_like(bn.running_mean))
    assert pre.classifier[-1].out_features == 2


# -------------------------------------------------------------- congelamiento

@pytest.mark.parametrize("name", Mo.MODEL_NAMES)
def test_freeze_backbone_trains_only_head(name):
    model = build_model(name, pretrained=False, freeze="backbone")
    assert _trainable(model) == _params(head_module(model, name))
    total, trainable = count_params(model)
    assert 0 < trainable < total


@pytest.mark.parametrize("name", Mo.MODEL_NAMES)
def test_freeze_partial_trains_last_30pct_blocks_and_head(name):
    model = build_model(name, pretrained=False, freeze="partial")
    blocks = backbone_blocks(model, name)
    n_last = max(1, math.ceil(0.3 * len(blocks)))
    expected = _params(head_module(model, name))
    for b in blocks[-n_last:]:
        expected |= _params(b)
    assert _trainable(model) == expected
    frozen_blocks = blocks[:-n_last]
    assert frozen_blocks, "debe haber bloques congelados"
    for b in frozen_blocks:
        assert not any(p.requires_grad for p in b.parameters())


def test_freeze_partial_block_counts_by_hand():
    """mobilenet_v3_small: 13 bloques → ceil(3.9) = 4; resnet18: stem + 8 → ceil(2.7) = 3."""
    mnv3 = build_model("mobilenet_v3_small", pretrained=False, freeze="partial")
    flags = [any(p.requires_grad for p in b.parameters()) for b in mnv3.features]
    assert flags == [False] * 9 + [True] * 4
    r18 = build_model("resnet18", pretrained=False, freeze="partial")
    assert not any(p.requires_grad for p in r18.layer3[0].parameters())
    assert all(p.requires_grad for p in r18.layer3[1].parameters())
    assert all(p.requires_grad for p in r18.layer4.parameters())
    assert not any(p.requires_grad for p in r18.conv1.parameters())


def test_partial_has_more_trainable_than_backbone():
    for name in ("mobilenet_v3_small", "resnet18", "malexnet"):
        n_b = count_params(build_model(name, pretrained=False, freeze="backbone"))[1]
        n_p = count_params(build_model(name, pretrained=False, freeze="partial"))[1]
        n_n = count_params(build_model(name, pretrained=False, freeze="none"))[1]
        assert n_b < n_p < n_n


def test_apply_freeze_none_unfreezes_everything():
    model = build_model("resnet18", pretrained=False, freeze="backbone")
    apply_freeze(model, "resnet18", "none")
    assert all(p.requires_grad for p in model.parameters())


def _bns(module: nn.Module) -> list[nn.Module]:
    return [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def test_set_train_mode_keeps_frozen_bn_in_eval():
    model = build_model("mobilenet_v3_small", pretrained=False, freeze="backbone")
    set_train_mode(model)
    assert model.training
    feature_bns = _bns(model.features)
    assert feature_bns and all(not bn.training for bn in feature_bns)
    assert all(m.training for m in model.classifier.modules())  # dropout de la cabeza activo


def test_set_train_mode_frozen_bn_stats_do_not_change():
    model = build_model("mobilenet_v3_small", pretrained=False, freeze="backbone")
    set_train_mode(model)
    before = [bn.running_mean.clone() for bn in _bns(model.features)]
    with torch.no_grad():
        model(torch.randn(4, 3, 64, 64) * 3 + 1)
    after = [bn.running_mean for bn in _bns(model.features)]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_set_train_mode_partial_and_none():
    part = set_train_mode(build_model("mobilenet_v3_small", pretrained=False, freeze="partial"))
    for i, block in enumerate(part.features):
        for bn in _bns(block):
            assert bn.training == (i >= 9)
    full = set_train_mode(build_model("mobilenet_v3_small", pretrained=False, freeze="none"))
    assert all(bn.training for bn in _bns(full))


# ------------------------------------------------------------ LRN en fp32

def test_fp32_lrn_matches_reference_on_cpu():
    x = torch.randn(2, 16, 9, 9)
    ref = nn.LocalResponseNorm(5)(x)
    out = FP32LocalResponseNorm(5)(x)
    assert out.dtype == x.dtype
    assert torch.allclose(out, ref)


@pytest.mark.skipif(not HAS_CUDA, reason="requiere CUDA")
def test_fp32_lrn_finite_under_autocast_with_large_inputs():
    """Con entradas grandes la suma de cuadrados no cabe en fp16 (máx. 65,504)."""
    x = torch.full((2, 16, 13, 13), 400.0, device="cuda").half()
    lrn = FP32LocalResponseNorm(5).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = lrn(x)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    ref = nn.LocalResponseNorm(5)(x.float())
    assert torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not HAS_CUDA, reason="requiere CUDA")
def test_malexnet_forward_finite_under_autocast_with_large_inputs():
    torch.manual_seed(0)
    model = build_model("malexnet", pretrained=False).cuda().eval()
    x = torch.randn(4, 3, 224, 224, device="cuda") * 200
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(x)
    assert out.shape == (4, 2)
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not HAS_CUDA, reason="requiere CUDA")
def test_malexnet_backward_finite_under_autocast():
    torch.manual_seed(0)
    model = build_model("malexnet", pretrained=False).cuda()
    x = torch.randn(4, 3, 224, 224, device="cuda") * 50
    y = torch.tensor([0, 1, 0, 1], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
