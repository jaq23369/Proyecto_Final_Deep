"""Pruebas de transforms.py: normalización, forma, aumento y paridad numpy/torchvision."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cupo.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transforms, preprocess_numpy

torch = pytest.importorskip("torch")


def _random_image(mode: str, size: tuple[int, int], seed: int) -> Image.Image:
    """Imagen sintética con ruido (ancho, alto) en el modo pedido."""
    rng = np.random.default_rng(seed)
    w, h = size
    if mode == "L":
        arr = rng.integers(0, 256, (h, w), dtype=np.uint8)
    elif mode == "RGBA":
        arr = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    else:
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


# --------------------------------------------------------- valores a mano

def test_imagenet_constants():
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)


def test_preprocess_numpy_constant_color_hand_computed():
    # Color uniforme (255, 0, 51): el resize no cambia los valores
    img = Image.new("RGB", (20, 10), (255, 0, 51))
    out = preprocess_numpy(img, 8)
    expected = [
        (1.0 - 0.485) / 0.229,   # 2.2489...
        (0.0 - 0.456) / 0.224,   # -2.0357...
        (0.2 - 0.406) / 0.225,   # -0.9155...
    ]
    for c in range(3):
        np.testing.assert_allclose(out[0, c], expected[c], atol=1e-5)


def test_preprocess_numpy_gray_constant_hand_computed():
    # Gris 128 → los tres canales valen 128/255
    img = Image.new("L", (12, 12), 128)
    out = preprocess_numpy(img, 4)
    v = 128 / 255
    for c in range(3):
        np.testing.assert_allclose(out[0, c], (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c], atol=1e-5)


# ---------------------------------------------------------- forma y dtype

@pytest.mark.parametrize("size", [32, 64, 224])
def test_preprocess_numpy_shape_dtype(size):
    out = preprocess_numpy(_random_image("RGB", (150, 150), 0), size)
    assert out.shape == (1, 3, size, size)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_preprocess_numpy_does_not_modify_input():
    img = _random_image("L", (30, 20), 1)
    before = np.asarray(img).copy()
    preprocess_numpy(img, 16)
    assert img.mode == "L"
    np.testing.assert_array_equal(np.asarray(img), before)


# --------------------------------------------- paridad numpy vs torchvision

@pytest.mark.parametrize(
    "mode,size,input_size",
    [
        ("RGB", (150, 150), 224),   # patch típico, ampliación
        ("RGB", (150, 150), 64),    # reducción
        ("RGB", (97, 53), 32),      # no cuadrada
        ("L", (150, 150), 224),     # escala de grises
        ("L", (80, 120), 48),       # gris no cuadrada
        ("RGBA", (64, 64), 32),     # con canal alfa
    ],
)
@pytest.mark.parametrize("augment", [False, True])
def test_preprocess_numpy_matches_eval_transform(mode, size, input_size, augment):
    img = _random_image(mode, size, seed=size[0] * 1000 + size[1] + input_size)
    # Igual que ParkingDataset: torchvision recibe la imagen ya en RGB
    ref = build_transforms(input_size, train=False, augment=augment)(img.convert("RGB"))
    ref = ref.unsqueeze(0).numpy()
    out = preprocess_numpy(img, input_size)
    assert out.shape == ref.shape
    assert float(np.abs(out - ref).max()) < 1e-5


# ------------------------------------------------------- build_transforms

@pytest.mark.parametrize("input_size", [32, 224])
def test_build_transforms_train_augment_shape(input_size):
    tf = build_transforms(input_size, train=True, augment=True)
    img = _random_image("RGB", (150, 150), 3)
    torch.manual_seed(0)
    for _ in range(5):
        x = tf(img)
        assert tuple(x.shape) == (3, input_size, input_size)
        assert x.dtype == torch.float32
        assert torch.isfinite(x).all()


def test_build_transforms_augment_is_random_in_train():
    tf = build_transforms(32, train=True, augment=True)
    img = _random_image("RGB", (150, 150), 4)
    torch.manual_seed(0)
    outs = [tf(img) for _ in range(6)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])


def test_build_transforms_augment_reproducible_with_seed():
    tf = build_transforms(32, train=True, augment=True)
    img = _random_image("RGB", (150, 150), 5)
    torch.manual_seed(11)
    a = [tf(img) for _ in range(3)]
    torch.manual_seed(11)
    b = [tf(img) for _ in range(3)]
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("train,augment", [(False, False), (False, True), (True, False)])
def test_build_transforms_no_augment_is_deterministic(train, augment):
    tf = build_transforms(32, train=train, augment=augment)
    img = _random_image("RGB", (150, 150), 6)
    torch.manual_seed(0)
    a = tf(img)
    torch.manual_seed(999)
    b = tf(img)
    assert torch.equal(a, b)


def test_build_transforms_augment_only_when_train_and_augment():
    from torchvision import transforms as T

    def names(tf):
        return {type(op).__name__ for op in tf.transforms}

    aug = names(build_transforms(32, train=True, augment=True))
    for kind in ("ColorJitter", "RandomAffine", "RandomApply", "RandomHorizontalFlip", "RandomErasing"):
        assert kind in aug
    for train, augment in [(False, False), (False, True), (True, False)]:
        plain = build_transforms(32, train=train, augment=augment)
        assert [type(op) for op in plain.transforms] == [T.Resize, T.ToTensor, T.Normalize]


def test_build_transforms_spec_parameters():
    from torchvision import transforms as T

    tf = build_transforms(64, train=True, augment=True)
    ops = {type(op).__name__: op for op in tf.transforms}
    resize = ops["Resize"]
    assert tuple(resize.size) == (64, 64)
    assert resize.interpolation == T.InterpolationMode.BILINEAR
    norm = ops["Normalize"]
    assert tuple(norm.mean) == IMAGENET_MEAN and tuple(norm.std) == IMAGENET_STD
    cj = ops["ColorJitter"]
    assert cj.brightness == (0.6, 1.4) and cj.contrast == (0.6, 1.4)
    assert cj.saturation == (0.7, 1.3) and cj.hue == (-0.02, 0.02)
    aff = ops["RandomAffine"]
    assert tuple(aff.degrees) == (-5.0, 5.0)
    assert tuple(aff.translate) == (0.05, 0.05) and tuple(aff.scale) == (0.9, 1.1)
    assert ops["RandomApply"].p == pytest.approx(0.2)
    blur = ops["RandomApply"].transforms[0]
    assert isinstance(blur, T.GaussianBlur)
    assert tuple(blur.kernel_size) == (5, 5)
    assert tuple(blur.sigma) == pytest.approx((0.07, 1.0))
    erasing = ops["RandomErasing"]
    assert erasing.p == pytest.approx(0.25)
    assert tuple(erasing.scale) == (0.02, 0.08) and erasing.value == 0


def test_build_transforms_augment_order():
    """Aumento sobre el patch original (antes del Resize); RandomErasing al final."""
    tf = build_transforms(224, train=True, augment=True)
    kinds = [type(op).__name__ for op in tf.transforms]
    assert kinds == [
        "ColorJitter", "RandomAffine", "RandomApply", "RandomHorizontalFlip",
        "Resize", "ToTensor", "Normalize", "RandomErasing",
    ]
    resize = kinds.index("Resize")
    for aug in ("ColorJitter", "RandomAffine", "RandomApply", "RandomHorizontalFlip"):
        assert kinds.index(aug) < resize
    assert kinds.index("RandomErasing") > kinds.index("Normalize")


@pytest.mark.parametrize("size", [(126, 150), (150, 126), (126, 126), (133, 141), (150, 150)])
@pytest.mark.parametrize("input_size", [64, 224])
def test_build_transforms_augment_non_square_inputs(size, input_size):
    # El aumento corre al tamaño original; la salida siempre es (3, s, s)
    tf = build_transforms(input_size, train=True, augment=True)
    img = _random_image("RGB", size, seed=size[0] + size[1])
    torch.manual_seed(1)
    for _ in range(4):
        x = tf(img)
        assert tuple(x.shape) == (3, input_size, input_size)
        assert x.dtype == torch.float32 and torch.isfinite(x).all()


def test_build_transforms_eval_unchanged_by_augment_flag():
    # Sin aumento, el pipeline sigue siendo Resize → ToTensor → Normalize
    img = _random_image("RGB", (126, 150), 9)
    a = build_transforms(32, train=False, augment=True)(img)
    b = build_transforms(32, train=True, augment=False)(img)
    assert torch.equal(a, b)
    np.testing.assert_allclose(preprocess_numpy(img, 32)[0], a.numpy(), atol=1e-5)
