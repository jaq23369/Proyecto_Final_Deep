"""Pruebas de scripts/download_data.py sin red (urlopen simulado) y prueba de humo real."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tarfile
import urllib.error
import zipfile
from pathlib import Path

import pytest

from cupo import paths

SCRIPT = paths.repo_root() / "scripts" / "download_data.py"
HAS_DATA = paths.metadata_csv().is_file() and paths.split_file("train").is_file()


def _load_module():
    spec = importlib.util.spec_from_file_location("download_data_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses necesita el módulo registrado en sys.modules
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


dd = _load_module()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Cualquier acceso a la red no simulado hace fallar la prueba."""
    def boom(*args, **kwargs):
        raise AssertionError("acceso a la red no simulado")

    monkeypatch.setattr(dd.urllib.request, "urlopen", boom)


# ------------------------------------------------------------- utilidades

class FakeResponse:
    """Respuesta HTTP mínima: status, headers y read(n)."""

    def __init__(self, body: bytes, status: int = 200, content_length: int | None = -1):
        self._buf = io.BytesIO(body)
        self.status = status
        length = len(body) if content_length == -1 else content_length
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self.closed = False

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "error", {}, None)


class Recorder:
    """urlopen simulado: devuelve o lanza en orden y guarda los headers de cada llamada."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out

    def range_header(self, i: int):
        return self.requests[i].get_header("Range")


# ---------------------------------------------------------------- is_ready

def test_is_ready_plain_file(tmp_path):
    asset = dd.Asset("CNRPark+EXT.csv", None)
    assert not dd.is_ready(asset, tmp_path)
    (tmp_path / "CNRPark+EXT.csv").write_text("x", encoding="utf-8")
    assert dd.is_ready(asset, tmp_path)


def test_is_ready_requires_sentinel(tmp_path):
    asset = dd.Asset("CNR-EXT-Patches-150x150.zip", "CNR-EXT")
    assert not dd.is_ready(asset, tmp_path)
    target = tmp_path / "CNR-EXT"
    (target / "PATCHES").mkdir(parents=True)
    (tmp_path / asset.filename).write_bytes(b"zip")
    # Carpeta con contenido y archivo presente, pero sin .complete → no está lista
    assert not dd.is_ready(asset, tmp_path)
    (target / dd.SENTINEL).mkdir()  # una carpeta con ese nombre no cuenta
    assert not dd.is_ready(asset, tmp_path)
    (target / dd.SENTINEL).rmdir()
    (target / dd.SENTINEL).write_text("ok\n", encoding="utf-8")
    assert dd.is_ready(asset, tmp_path)


def test_assets_and_expected_counts_match_plan():
    names = {a.filename: a.extract_to for a in dd.ASSETS}
    assert names == {
        "CNRPark+EXT.csv": None,
        "CNRPark-Patches-150x150.zip": "CNRPark",
        "CNR-EXT-Patches-150x150.zip": "CNR-EXT",
        "CNR-EXT_FULL_IMAGE_1000x750.tar": "CNR-EXT_FULL",
    }
    assert dd.EXPECTED["train"] == 94_493 and dd.EXPECTED["val"] == 18_647
    assert dd.EXPECTED["test"] == 31_825 and dd.EXPECTED["ood"] == 12_584
    assert dd.EXPECTED["csv_rows"] == 157_549 and dd.EXPECTED["ext_unique_slots"] == 164
    assert dd.EXPECTED["full_frames"] == 4_278 and dd.EXPECTED["camera_csvs"] == 9
    assert dd.EXPECTED_BUSY_PCT == {"train": 50.4, "val": 71.9, "test": 57.4}


# ------------------------------------------------------- manejo del 416

def test_open_download_416_part_complete(monkeypatch, tmp_path):
    dest = tmp_path / "f.zip"
    part = tmp_path / "f.zip.part"
    part.write_bytes(b"0123456789")
    rec = Recorder(_http_error(416))
    monkeypatch.setattr(dd.urllib.request, "urlopen", rec)
    monkeypatch.setattr(dd, "remote_size", lambda url: 10)
    assert dd._open_download("http://x/f.zip", part, dest) is None
    assert rec.range_header(0) == "bytes=10-"
    assert dest.read_bytes() == b"0123456789"
    assert not part.exists()


def test_open_download_416_part_wrong_size_restarts(monkeypatch, tmp_path):
    dest = tmp_path / "f.zip"
    part = tmp_path / "f.zip.part"
    part.write_bytes(b"basura-mas-larga-que-el-remoto")
    fresh = FakeResponse(b"nuevo")
    rec = Recorder(_http_error(416), fresh)
    monkeypatch.setattr(dd.urllib.request, "urlopen", rec)
    monkeypatch.setattr(dd, "remote_size", lambda url: 5)
    resp, start = dd._open_download("http://x/f.zip", part, dest)
    assert resp is fresh and start == 0
    assert not part.exists() and not dest.exists()
    assert rec.range_header(0) is not None and rec.range_header(1) is None


def test_open_download_416_unknown_remote_size_restarts(monkeypatch, tmp_path):
    dest = tmp_path / "f.zip"
    part = tmp_path / "f.zip.part"
    part.write_bytes(b"abc")
    rec = Recorder(_http_error(416), FakeResponse(b"abcdef"))
    monkeypatch.setattr(dd.urllib.request, "urlopen", rec)
    monkeypatch.setattr(dd, "remote_size", lambda url: None)
    _, start = dd._open_download("http://x/f.zip", part, dest)
    assert start == 0 and not part.exists()


def test_open_download_416_without_part_is_raised(monkeypatch, tmp_path):
    rec = Recorder(_http_error(416))
    monkeypatch.setattr(dd.urllib.request, "urlopen", rec)
    with pytest.raises(urllib.error.HTTPError):
        dd._open_download("http://x/f.zip", tmp_path / "f.zip.part", tmp_path / "f.zip")
    assert rec.range_header(0) is None


@pytest.mark.parametrize("code", [404, 500, 403])
def test_open_download_other_http_errors_are_raised(monkeypatch, tmp_path, code):
    part = tmp_path / "f.zip.part"
    part.write_bytes(b"abc")
    monkeypatch.setattr(dd.urllib.request, "urlopen", Recorder(_http_error(code)))
    with pytest.raises(urllib.error.HTTPError) as info:
        dd._open_download("http://x/f.zip", part, tmp_path / "f.zip")
    assert info.value.code == code
    assert part.read_bytes() == b"abc"  # no se toca el .part


# --------------------------------------------------------------- download

def test_download_fresh(monkeypatch, tmp_path):
    body = b"x" * (3 * dd.CHUNK + 17)
    monkeypatch.setattr(dd.urllib.request, "urlopen", Recorder(FakeResponse(body)))
    dest = tmp_path / "a.tar"
    dd.download("http://x/a.tar", dest)
    assert dest.read_bytes() == body
    assert not (tmp_path / "a.tar.part").exists()


def test_download_resumes_with_206(monkeypatch, tmp_path):
    dest = tmp_path / "a.tar"
    (tmp_path / "a.tar.part").write_bytes(b"hola ")
    rec = Recorder(FakeResponse(b"mundo", status=206))
    monkeypatch.setattr(dd.urllib.request, "urlopen", rec)
    dd.download("http://x/a.tar", dest)
    assert rec.range_header(0) == "bytes=5-"
    assert dest.read_bytes() == b"hola mundo"


def test_download_server_ignores_range(monkeypatch, tmp_path):
    # Si el servidor responde 200 a una petición con Range, se reescribe desde cero
    dest = tmp_path / "a.tar"
    (tmp_path / "a.tar.part").write_bytes(b"viejo")
    monkeypatch.setattr(dd.urllib.request, "urlopen", Recorder(FakeResponse(b"completo", status=200)))
    dd.download("http://x/a.tar", dest)
    assert dest.read_bytes() == b"completo"


def test_download_416_complete_part(monkeypatch, tmp_path, capsys):
    dest = tmp_path / "a.tar"
    (tmp_path / "a.tar.part").write_bytes(b"12345")
    monkeypatch.setattr(dd.urllib.request, "urlopen", Recorder(_http_error(416)))
    monkeypatch.setattr(dd, "remote_size", lambda url: 5)
    dd.download("http://x/a.tar", dest)
    assert dest.read_bytes() == b"12345"
    assert "ya estaba completo" in capsys.readouterr().out


def test_download_incomplete_raises_and_keeps_part(monkeypatch, tmp_path):
    dest = tmp_path / "a.tar"
    # Content-Length dice 100 pero llegan 10 bytes
    monkeypatch.setattr(dd.urllib.request, "urlopen",
                        Recorder(FakeResponse(b"0123456789", content_length=100)))
    with pytest.raises(OSError, match="incompleta"):
        dd.download("http://x/a.tar", dest)
    assert not dest.exists()
    assert (tmp_path / "a.tar.part").read_bytes() == b"0123456789"


def test_download_without_content_length(monkeypatch, tmp_path):
    dest = tmp_path / "a.tar"
    monkeypatch.setattr(dd.urllib.request, "urlopen",
                        Recorder(FakeResponse(b"abc", content_length=None)))
    dd.download("http://x/a.tar", dest)
    assert dest.read_bytes() == b"abc"


# ------------------------------------------------------ ensure_downloaded

def test_ensure_downloaded_skips_matching_size(monkeypatch, tmp_path):
    asset = dd.Asset("CNRPark+EXT.csv", None)
    (tmp_path / asset.filename).write_bytes(b"12345")
    monkeypatch.setattr(dd, "remote_size", lambda url: 5)
    monkeypatch.setattr(dd, "download", lambda *a: pytest.fail("no debía descargar"))
    assert dd.ensure_downloaded(asset, tmp_path) == tmp_path / asset.filename


def test_ensure_downloaded_redownloads_on_size_mismatch(monkeypatch, tmp_path):
    asset = dd.Asset("CNRPark+EXT.csv", None)
    (tmp_path / asset.filename).write_bytes(b"123")
    calls = []
    monkeypatch.setattr(dd, "remote_size", lambda url: 5)
    monkeypatch.setattr(dd, "download", lambda url, dest: calls.append((url, dest)))
    dd.ensure_downloaded(asset, tmp_path)
    assert calls == [(f"{dd.BASE_URL}/{asset.filename}", tmp_path / asset.filename)]
    assert not (tmp_path / asset.filename).exists()  # se borró el archivo viejo


# ------------------------------------------------------------ extracción

def test_extract_zip_writes_sentinel(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("LABELS/train.txt", "x 1\n")
    target = tmp_path / "out"
    dd.extract(archive, target)
    assert (target / "LABELS" / "train.txt").read_text() == "x 1\n"
    assert (target / dd.SENTINEL).is_file()
    assert dd.is_ready(dd.Asset("a.zip", "out"), tmp_path)


def test_extract_zip_rejects_unsafe_paths(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../fuera.txt", "x")
    target = tmp_path / "out"
    with pytest.raises(ValueError, match="insegura"):
        dd.extract(archive, target)
    assert not (tmp_path / "fuera.txt").exists()
    assert not (target / dd.SENTINEL).exists()


def test_extract_tar_writes_sentinel(tmp_path):
    src = tmp_path / "camera1.csv"
    src.write_text("SlotId,X,Y,W,H\n", encoding="utf-8")
    archive = tmp_path / "a.tar"
    with tarfile.open(archive, "w") as tf:
        tf.add(src, arcname="camera1.csv")
    target = tmp_path / "out"
    dd.extract(archive, target)
    assert (target / "camera1.csv").is_file() and (target / dd.SENTINEL).is_file()


def test_extract_unknown_format(tmp_path):
    archive = tmp_path / "a.rar"
    archive.write_bytes(b"x")
    with pytest.raises(ValueError):
        dd.extract(archive, tmp_path / "out")
    assert not (tmp_path / "out" / dd.SENTINEL).exists()


def test_prepare_skips_ready_assets(monkeypatch, tmp_path):
    for asset in dd.ASSETS:
        if asset.extract_to is None:
            (tmp_path / asset.filename).write_text("x", encoding="utf-8")
        else:
            (tmp_path / asset.extract_to).mkdir()
            (tmp_path / asset.extract_to / dd.SENTINEL).write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(dd, "ensure_downloaded", lambda *a: pytest.fail("no debía descargar"))
    dd.prepare(tmp_path)


# -------------------------------------------------------------- verify/CLI

def test_main_verify_only_on_empty_root_fails_without_network(tmp_path, capsys):
    assert dd.main(["--verify-only", "--data-root", str(tmp_path)]) == 1
    assert "Falta un archivo" in capsys.readouterr().out


def test_check_records_result():
    results: dict = {}
    dd._check(results, "x", 3, 3)
    dd._check(results, "y", 2, 3)
    assert results == {"x": {"value": 3, "expected": 3, "ok": True},
                       "y": {"value": 2, "expected": 3, "ok": False}}


def test_parse_args_defaults():
    args = dd.parse_args([])
    assert args.data_root is None and not args.verify_only and not args.delete_archives


@pytest.mark.data
@pytest.mark.skipif(not HAS_DATA, reason="dataset no disponible en DATA_ROOT")
def test_verify_only_smoke_real_data():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--verify-only"],
        cwd=paths.repo_root(), capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "FALLA" not in proc.stdout
    assert "Verificación: OK" in proc.stdout
