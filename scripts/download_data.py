"""Descarga, extrae y verifica CNRPark+EXT.

Uso:
    python scripts/download_data.py [--data-root RUTA] [--verify-only] [--delete-archives]

Los archivos vienen del release "archive" del repositorio de los autores
(fabiocarrara/deep-parking). Lo ya descargado o extraído se salta. Solo usa la
biblioteca estándar para poder correrlo antes de instalar las dependencias.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from http.client import HTTPResponse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cupo import paths  # noqa: E402

BASE_URL = "https://github.com/fabiocarrara/deep-parking/releases/download/archive"
CHUNK = 1 << 20  # 1 MiB
SENTINEL = ".complete"  # se escribe en la carpeta destino al terminar la extracción

# Conteos esperados (ver README); cualquier diferencia indica una descarga incompleta
EXPECTED = {
    "csv_rows": 157_549,
    "ext_unique_slots": 164,
    "train": 94_493,
    "val": 18_647,
    "test": 31_825,
    "ood": 12_584,
    "full_frames": 4_278,
    "camera_csvs": 9,
}
EXPECTED_BUSY_PCT = {"train": 50.4, "val": 71.9, "test": 57.4}


@dataclass(frozen=True)
class Asset:
    """Archivo del release y dónde queda su contenido bajo DATA_ROOT."""

    filename: str
    extract_to: str | None  # carpeta destino (None = no se extrae)


ASSETS: tuple[Asset, ...] = (
    Asset("CNRPark+EXT.csv", None),
    Asset("CNRPark-Patches-150x150.zip", "CNRPark"),           # contiene A/, B/
    Asset("CNR-EXT-Patches-150x150.zip", "CNR-EXT"),           # contiene PATCHES/, LABELS/
    Asset("CNR-EXT_FULL_IMAGE_1000x750.tar", "CNR-EXT_FULL"),  # FULL_IMAGE_1000x750/, camera<k>.csv
)


# ---------------------------------------------------------------- descarga

def remote_size(url: str) -> int | None:
    """Tamaño remoto (Content-Length) o None si no se puede obtener."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    return int(length) if length else None


def _print_progress(done: int, total: int | None, name: str) -> None:
    if total:
        pct = 100 * done / total
        msg = f"\r  {name}: {done / 1e6:,.1f}/{total / 1e6:,.1f} MB ({pct:5.1f} %)"
    else:
        msg = f"\r  {name}: {done / 1e6:,.1f} MB"
    print(msg, end="", flush=True)


def _open_download(url: str, part: Path, dest: Path) -> tuple[HTTPResponse, int] | None:
    """Abre la descarga reanudando desde ``part``.

    Devuelve ``(respuesta, inicio)`` o None si el ``.part`` ya estaba completo.

    Un HTTP 416 (rango fuera del archivo) significa que el ``.part`` ya está completo
    si su tamaño coincide con el remoto; si no, se borra y se empieza de cero.
    """
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60), start
    except urllib.error.HTTPError as exc:
        if exc.code != 416 or not start:
            raise
        if remote_size(url) == start:
            part.replace(dest)
            return None
        part.unlink()
        return urllib.request.urlopen(urllib.request.Request(url), timeout=60), 0


def download(url: str, dest: Path) -> None:
    """Descarga ``url`` en ``dest`` vía un archivo ``.part`` (reanuda si ya existe)."""
    part = dest.with_name(dest.name + ".part")
    opened = _open_download(url, part, dest)
    if opened is None:
        print(f"  {part.name} ya estaba completo")
        return
    response, start = opened
    with response as resp:
        resumed = resp.status == 206
        if not resumed:
            start = 0
        length = resp.headers.get("Content-Length")
        total = start + int(length) if length else None
        done = start
        with open(part, "ab" if resumed else "wb") as fh:
            while chunk := resp.read(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                _print_progress(done, total, dest.name)
    print()
    if total is not None and done != total:
        raise OSError(f"Descarga incompleta de {dest.name}: {done} de {total} bytes")
    part.replace(dest)


def ensure_downloaded(asset: Asset, root: Path) -> Path:
    """Descarga el archivo si no existe o si su tamaño no coincide con el remoto."""
    dest = root / asset.filename
    url = f"{BASE_URL}/{asset.filename}"
    if dest.exists():
        expected = remote_size(url)
        if expected is None or dest.stat().st_size == expected:
            print(f"  ya descargado: {asset.filename}")
            return dest
        print(f"  tamaño distinto al remoto, se descarga de nuevo: {asset.filename}")
        dest.unlink()
    print(f"  descargando {url}")
    download(url, dest)
    return dest


# --------------------------------------------------------------- extracción

def is_ready(asset: Asset, root: Path) -> bool:
    """Listo = archivo presente (si no se extrae) o centinela de extracción completa."""
    if asset.extract_to is None:
        return (root / asset.filename).is_file()
    return (root / asset.extract_to / SENTINEL).is_file()


def _extract_zip(archive: Path, target: Path) -> None:
    target_resolved = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            # Protección contra rutas que salen de la carpeta destino
            if not (target / name).resolve().is_relative_to(target_resolved):
                raise ValueError(f"Ruta insegura dentro de {archive.name}: {name}")
        zf.extractall(target)


def _extract_tar(archive: Path, target: Path) -> None:
    with tarfile.open(archive) as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(target, filter="data")
        else:
            target_resolved = target.resolve()
            for member in tf.getmembers():
                if not (target / member.name).resolve().is_relative_to(target_resolved):
                    raise ValueError(f"Ruta insegura dentro de {archive.name}: {member.name}")
            tf.extractall(target)


def extract(archive: Path, target: Path) -> None:
    print(f"  extrayendo {archive.name} -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        _extract_zip(archive, target)
    elif archive.suffix == ".tar":
        _extract_tar(archive, target)
    else:
        raise ValueError(f"Formato no soportado: {archive.name}")
    # Solo se marca como lista si la extracción terminó sin errores
    (target / SENTINEL).write_text("ok\n", encoding="utf-8")


def prepare(root: Path, delete_archives: bool = False) -> None:
    """Descarga y extrae lo que falte."""
    root.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS:
        if is_ready(asset, root):
            print(f"  listo, se salta: {asset.filename}")
            continue
        archive = ensure_downloaded(asset, root)
        if asset.extract_to is None:
            continue
        extract(archive, root / asset.extract_to)
        if delete_archives:
            archive.unlink()


# ------------------------------------------------------------- verificación

def _read_split(root: Path, name: str) -> list[tuple[str, int]]:
    lines = (root / "CNR-EXT" / "LABELS" / f"{name}.txt").read_text(encoding="utf-8").split("\n")
    rows = []
    for line in lines:
        if line.strip():
            rel, label = line.split()
            rows.append((rel, int(label)))
    return rows


def _check(results: dict, name: str, value, expected) -> None:
    ok = value == expected
    results[name] = {"value": value, "expected": expected, "ok": ok}
    print(f"  [{'OK' if ok else 'FALLA'}] {name}: {value} (esperado {expected})")


def _verify_csv(root: Path, results: dict) -> None:
    with open(root / "CNRPark+EXT.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing = sum(not (root / r["image_url"]).is_file() for r in rows)
    ext_slots = {r["slot_id"] for r in rows if r["camera"] not in ("A", "B")}
    ood_rows = sum(r["camera"] in ("A", "B") for r in rows)
    _check(results, "csv_rows", len(rows), EXPECTED["csv_rows"])
    _check(results, "csv_missing_files", missing, 0)
    _check(results, "ext_unique_slots", len(ext_slots), EXPECTED["ext_unique_slots"])
    _check(results, "ood", ood_rows, EXPECTED["ood"])


def _verify_splits(root: Path, results: dict) -> None:
    patches = root / "CNR-EXT" / "PATCHES"
    dates: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        rows = _read_split(root, name)
        missing = sum(not (patches / rel).is_file() for rel, _ in rows)
        busy_pct = round(100 * sum(lbl for _, lbl in rows) / max(1, len(rows)), 1)
        dates[name] = {rel.split("/")[1] for rel, _ in rows}
        _check(results, name, len(rows), EXPECTED[name])
        _check(results, f"{name}_missing_files", missing, 0)
        _check(results, f"{name}_busy_pct", busy_pct, EXPECTED_BUSY_PCT[name])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        _check(results, f"dates_{a}_and_{b}", len(dates[a] & dates[b]), 0)


def _verify_full(root: Path, results: dict) -> None:
    full = root / "CNR-EXT_FULL"
    n_frames = sum(1 for _ in (full / "FULL_IMAGE_1000x750").glob("*/*/camera*/*.jpg"))
    n_csvs = sum((full / f"camera{k}.csv").is_file() for k in range(1, 10))
    n_ood_files = sum(1 for _ in (root / "CNRPark").glob("[AB]/*/*.jpg"))
    _check(results, "full_frames", n_frames, EXPECTED["full_frames"])
    _check(results, "camera_csvs", n_csvs, EXPECTED["camera_csvs"])
    _check(results, "ood_files_on_disk", n_ood_files, EXPECTED["ood"])


def verify(root: Path) -> dict:
    """Comprueba conteos, archivos faltantes y que las particiones no compartan fechas.

    Devuelve ``{check: {value, expected, ok}}`` más la llave ``ok`` global.
    """
    print(f"Verificando {root}")
    results: dict = {}
    _verify_csv(root, results)
    _verify_splits(root, results)
    _verify_full(root, results)
    results["ok"] = all(r["ok"] for r in results.values())
    print("Verificación:", "OK" if results["ok"] else "FALLA")
    return results


# --------------------------------------------------------------------- CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga y verifica CNRPark+EXT.")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Carpeta de datos (por defecto $DATA_ROOT o ../Data).")
    parser.add_argument("--verify-only", action="store_true", help="Solo verificar, sin descargar.")
    parser.add_argument("--delete-archives", action="store_true",
                        help="Borrar .zip/.tar después de extraerlos.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = (args.data_root or paths.data_root()).expanduser().resolve()
    if not args.verify_only:
        prepare(root, delete_archives=args.delete_archives)
    try:
        ok = verify(root)["ok"]
    except FileNotFoundError as exc:
        print(f"Falta un archivo: {exc}")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
