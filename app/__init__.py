"""App web de Cupo Seguro (FastAPI + ONNX Runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

# Permite arrancar la app aunque el paquete ``cupo`` no esté instalado con ``pip install -e .``
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    try:
        import cupo  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(_SRC))
