"""El README describe el repo tal como está.

- Cada ``python -m cupo.X`` y ``python scripts/Y.py`` citado existe y su ``--help`` termina en 0.
- Cada flag que el README usa con un comando aparece en el ``--help`` de ese comando.
- Los archivos y carpetas citados como parte del repo existen.
- Los conteos y porcentajes de las particiones coinciden con ``results/eda/eda_stats.json`` y con
  los conteos que verifica ``scripts/download_data.py``.
- Los enlaces internos (``#anclas``) apuntan a encabezados reales.

No ejecuta entrenamientos ni evaluaciones: solo ``--help`` y lecturas estáticas.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
TEXT = README.read_text(encoding="utf-8")
LINES = TEXT.splitlines()

MODULE_RX = re.compile(r"python -m (cupo\.\w+)")
SCRIPT_RX = re.compile(r"python (scripts/\w+\.py)")
COMMAND_RX = re.compile(r"python (?:-m (cupo\.\w+)|(scripts/\w+\.py))")
FLAG_RX = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")
INLINE_RX = re.compile(r"`([^`\n]+)`")

# Flags que el README menciona en prosa (fuera del comando) con cada comando.
PROSE_FLAGS: dict[str, set[str]] = {
    "scripts/download_data.py": {"--verify-only", "--data-root"},
    "cupo.train": {"--seed", "--subset", "--epochs", "--num-workers", "--n-boot", "--run-name",
                   "--runs-dir"},
    "scripts/train_classical.py": {"--config", "--subset", "--seed", "--run-name", "--runs-dir",
                                   "--n-boot"},
    "cupo.evaluate": {"--run", "--split", "--allow-test", "--batch-size", "--num-workers",
                      "--n-boot", "--device"},
    "cupo.calibrate": {"--run", "--n-boot"},
    "scripts/results_table.py": {"--runs-dir", "--out", "--include-smoke"},
    "scripts/make_figures.py": {"--runs-dir", "--include-smoke", "--out", "--with-table",
                                "--table-out"},
    "scripts/eda.py": {"--sample-per-split", "--seed", "--out", "--full-hash"},
    "cupo.export_onnx": {"--run", "--out", "--opset", "--n-check"},
    "cupo.benchmark": {"--onnx", "--run", "--n", "--n-frame", "--out", "--no-gpu"},
    "scripts/gradcam.py": {"--run", "--split", "--n", "--out", "--device"},
    "scripts/build_gallery.py": {"--out"},
}
# Flags de herramientas externas (pip, uvicorn) que también aparecen en el README.
EXTERNAL_FLAGS = {"--no-deps", "--index-url", "--upgrade", "--port"}


# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------

def _key(m: re.Match) -> str:
    return m.group(1) or m.group(2)


def _segments(text: str) -> list[tuple[str, str]]:
    """(comando, texto que le sigue hasta el siguiente comando) dentro de un fragmento."""
    matches = list(COMMAND_RX.finditer(text))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((_key(m), text[m.end():end]))
    return out


def command_flags() -> dict[str, set[str]]:
    """Flags usados con cada comando: en bloques de código, la línea (y sus continuaciones
    ``[--x]``); fuera de ellos, solo dentro del mismo span de código en línea."""
    flags: dict[str, set[str]] = {}
    in_fence = False
    last_cmd: str | None = None
    for line in LINES:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            last_cmd = None
            continue
        if in_fence:
            segs = _segments(line)
            if segs:
                for cmd, rest in segs:
                    flags.setdefault(cmd, set()).update(FLAG_RX.findall(rest))
                last_cmd = segs[-1][0]
            elif last_cmd and re.match(r"^\s+\[--", line):
                flags[last_cmd].update(FLAG_RX.findall(line))
            else:
                last_cmd = None
        else:
            for span in INLINE_RX.findall(line):
                for cmd, rest in _segments(span):
                    flags.setdefault(cmd, set()).update(FLAG_RX.findall(rest))
    return flags


def all_code_flags() -> set[str]:
    """Todos los flags que aparecen en código (bloques y spans en línea) del README."""
    found: set[str] = set()
    in_fence = False
    for line in LINES:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            found.update(FLAG_RX.findall(line))
        else:
            for span in INLINE_RX.findall(line):
                found.update(FLAG_RX.findall(span))
    return found


FLAGS = command_flags()
COMMANDS = sorted(set(MODULE_RX.findall(TEXT)) | set(SCRIPT_RX.findall(TEXT)))


def _python() -> str:
    for cand in (REPO / ".venv" / "Scripts" / "python.exe", REPO / ".venv" / "bin" / "python"):
        if cand.is_file():
            return str(cand)
    return sys.executable


@functools.lru_cache(maxsize=None)
def run_help(cmd: str) -> subprocess.CompletedProcess:
    args = ["-m", cmd] if cmd.startswith("cupo.") else [cmd]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO / "src"), env.get("PYTHONPATH")]))
    return subprocess.run([_python(), *args, "--help"], cwd=REPO, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env, timeout=300)


def _source(cmd: str) -> Path:
    if cmd.startswith("cupo."):
        return REPO / "src" / "cupo" / (cmd.split(".", 1)[1] + ".py")
    return REPO / cmd


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def test_readme_lists_expected_commands():
    expected = {
        "cupo.train", "cupo.evaluate", "cupo.calibrate", "cupo.export_onnx", "cupo.benchmark",
        "scripts/download_data.py", "scripts/eda.py", "scripts/train_classical.py",
        "scripts/results_table.py", "scripts/make_figures.py", "scripts/gradcam.py",
        "scripts/build_gallery.py",
    }
    assert expected <= set(COMMANDS), f"faltan en el README: {expected - set(COMMANDS)}"


@pytest.mark.parametrize("cmd", COMMANDS)
def test_command_source_exists(cmd):
    src = _source(cmd)
    assert src.is_file(), f"{cmd}: no existe {src.relative_to(REPO)}"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    has_main_guard = any(
        isinstance(node, ast.If) and "__main__" in ast.unparse(node.test) for node in tree.body
    )
    assert has_main_guard, f"{cmd}: sin bloque if __name__ == '__main__'"


@pytest.mark.parametrize("cmd", COMMANDS)
def test_command_help_exits_zero(cmd):
    res = run_help(cmd)
    assert res.returncode == 0, f"{cmd} --help → código {res.returncode}\n{res.stderr[-2000:]}"
    assert "usage" in res.stdout.lower() or "uso" in res.stdout.lower(), res.stdout[:500]


@pytest.mark.parametrize("cmd", COMMANDS)
def test_readme_flags_in_help(cmd):
    used = FLAGS.get(cmd, set()) | PROSE_FLAGS.get(cmd, set())
    res = run_help(cmd)
    assert res.returncode == 0, res.stderr[-2000:]
    offered = set(FLAG_RX.findall(res.stdout))
    missing = sorted(used - offered)
    assert not missing, f"{cmd}: el README usa {missing}, que no aparecen en --help"


def test_prose_flags_are_in_readme():
    """Los flags en prosa listados arriba están de verdad en el README (en código)."""
    code_flags = all_code_flags()
    missing = {cmd: sorted(f for f in fl if f not in code_flags) for cmd, fl in PROSE_FLAGS.items()}
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, f"flags listados que el README ya no menciona: {missing}"


def test_every_readme_flag_is_attributed():
    """Cada flag del README pertenece a algún comando del repo o a una herramienta externa."""
    attributed = set().union(*FLAGS.values(), *PROSE_FLAGS.values(), EXTERNAL_FLAGS)
    orphan = sorted(all_code_flags() - attributed)
    assert not orphan, f"flags del README sin comando asociado (agregar a PROSE_FLAGS): {orphan}"


def test_usage_block_of_train_matches_help():
    """El bloque 'Opciones de cupo.train' lista exactamente las opciones del --help."""
    start = TEXT.index("### Opciones de `cupo.train`")
    block = TEXT[start:].split("```")[1]
    readme_opts = set(FLAG_RX.findall(block))
    res = run_help("cupo.train")
    help_opts = set(FLAG_RX.findall(res.stdout)) - {"--help"}
    assert readme_opts == help_opts, (
        f"solo en README: {sorted(readme_opts - help_opts)}; solo en --help: {sorted(help_opts - readme_opts)}"
    )


def test_test_guard_flag_documented_and_enforced():
    """El README promete que --split test|ood exige --allow-test; el --help lo ofrece."""
    assert "--allow-test" in FLAGS.get("cupo.evaluate", set())
    out = run_help("cupo.evaluate").stdout
    assert "--allow-test" in out
    for split in ("test", "ood"):
        assert re.search(rf"\b{split}\b", out), f"--help de evaluate no menciona {split}"


def test_readme_snippet_imports_resolve():
    """El comando de una línea que busca el mejor run usa nombres que existen."""
    tree = ast.parse((REPO / "scripts" / "results_table.py").read_text(encoding="utf-8"))
    defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"load_records", "best_record"} <= defs
    from cupo import paths
    assert callable(getattr(paths, "runs_dir", None))


def test_readme_python_snippet_runs():
    """El ejemplo de sensibilidad del README corre tal cual (solo modelo de valor, sin datos)."""
    blocks = re.findall(r"```python\n(.*?)```", TEXT, flags=re.S)
    snippet = next(b for b in blocks if "sensitivity" in b)
    ns: dict = {}
    exec(compile(snippet.replace("print(", "_ = ("), "README", "exec"), ns)
    tabla = ns["tabla"]
    assert "delta" in tabla.columns and len(tabla) > 0
    from cupo import value_model
    assert callable(getattr(value_model, "value_curve", None))


# ---------------------------------------------------------------------------
# Archivos y carpetas
# ---------------------------------------------------------------------------

def _tree_block() -> str:
    start = TEXT.index("## Estructura del repositorio")
    return TEXT[start:].split("```")[1]


def test_tree_entries_exist():
    """Los archivos del árbol de 'Estructura del repositorio' existen donde se dice."""
    block = _tree_block()
    names = set(re.findall(r"\b[\w-]+\.(?:py|yaml|toml|txt)\b", block))
    search_dirs = [REPO, REPO / "configs", REPO / "configs" / "value", REPO / "src" / "cupo",
                   REPO / "scripts", REPO / "app"]
    missing = sorted(n for n in names if not any((d / n).is_file() for d in search_dirs))
    assert not missing, f"citados en el árbol pero inexistentes: {missing}"
    # Cada .py bajo src/cupo/ en el árbol
    src_part = block[block.index("src/cupo/"):block.index("scripts/")]
    for name in re.findall(r"\b(\w+)\.py\b", src_part):
        assert (REPO / "src" / "cupo" / f"{name}.py").is_file(), f"src/cupo/{name}.py"
    scripts_part = block[block.index("scripts/"):block.index("app/")]
    for name in re.findall(r"\b(\w+)\.py\b", scripts_part):
        assert (REPO / "scripts" / f"{name}.py").is_file(), f"scripts/{name}.py"


@pytest.mark.parametrize("d", ["configs", "configs/value", "src/cupo", "scripts", "app",
                               "app/static", "models", "results", "results/eda", "docs", "tests"])
def test_tree_dirs_exist(d):
    assert (REPO / d).is_dir(), d


def test_src_cupo_fully_documented():
    """Todo módulo de src/cupo/ aparece en el árbol del README."""
    block = _tree_block()
    mods = {p.name for p in (REPO / "src" / "cupo").glob("*.py") if p.name != "__init__.py"}
    missing = sorted(m for m in mods if m not in block)
    assert not missing, f"módulos sin mencionar en el árbol: {missing}"


def test_scripts_fully_documented():
    block = _tree_block()
    scripts = {p.name for p in (REPO / "scripts").glob("*.py")}
    missing = sorted(s for s in scripts if s not in block)
    assert not missing, f"scripts sin mencionar en el árbol: {missing}"


def test_cited_paths_exist():
    """Rutas del repo citadas en el texto (configs, src, scripts, app, results/eda)."""
    cited = set(re.findall(r"\b((?:configs|src/cupo|scripts|app|results/eda)/[\w/.-]*\w)", TEXT))
    # Salidas que se generan al correr (no están versionadas) y comodines
    generated = {"app/gallery.json", "app/feedback.jsonl"}
    missing = []
    for rel in sorted(cited):
        if rel in generated or "*" in rel or "<" in rel:
            continue
        if not (REPO / rel).exists():
            missing.append(rel)
    assert not missing, f"rutas citadas que no existen: {missing}"


@pytest.mark.parametrize("name", re.findall(r"configs/([\w/]+\.yaml)", TEXT))
def test_cited_configs_load(name):
    path = REPO / "configs" / name
    assert path.is_file(), name
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data


def test_run_configs_name_matches_iteration_table():
    """Cada config de la tabla de iteraciones tiene ``name`` igual al archivo."""
    for name in re.findall(r"`configs/([bk]\d_\w+)\.yaml`", TEXT):
        cfg = yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
        assert cfg["name"] == name, f"{name}.yaml tiene name={cfg['name']!r}"


def test_app_entrypoint_and_endpoints():
    """``uvicorn app.main:app`` y los endpoints listados existen en app/main.py."""
    src = (REPO / "app" / "main.py").read_text(encoding="utf-8")
    assert re.search(r"^app\s*=\s*FastAPI\(", src, flags=re.M), "app/main.py no define app"
    def norm(path: str) -> str:
        return re.sub(r"\{[^}]*\}", "{}", path)

    routes = {(verb.upper(), norm(path))
              for verb, path in re.findall(r"@app\.(get|post)\(\"([^\"]+)\"", src)}
    readme = re.findall(r"`(GET|POST) (/[^`]*)`", TEXT)
    assert readme, "el README no lista endpoints"
    missing = [f"{m} {p}" for m, p in readme if (m, norm(p)) not in routes]
    assert not missing, f"endpoints del README que no están en app/main.py: {missing}"


def test_env_vars_used_by_code():
    app_src = (REPO / "app" / "main.py").read_text(encoding="utf-8") + \
        (REPO / "app" / "inference.py").read_text(encoding="utf-8")
    for var in ("MODEL_PATH", "FEEDBACK_PATH"):
        assert f'"{var}"' in app_src, var
    assert "DATA_ROOT" in (REPO / "src" / "cupo" / "paths.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Números
# ---------------------------------------------------------------------------

def _int(s: str) -> int:
    return int(s.replace(",", ""))


@functools.lru_cache(maxsize=None)
def eda_stats() -> dict:
    return json.loads((REPO / "results" / "eda" / "eda_stats.json").read_text(encoding="utf-8"))


def _partition_table() -> dict[str, tuple[int, float]]:
    rows = {}
    for key, label in (("train", "train"), ("val", "val"), ("test", "test"), ("ood", "Test-OOD")):
        m = re.search(rf"^\| {re.escape(label)} \|[^|]*\| ([\d,]+) \| ([\d.]+) % \|$", TEXT, flags=re.M)
        assert m, f"fila {label} no encontrada en la tabla de particiones"
        rows[key] = (_int(m.group(1)), float(m.group(2)))
    return rows


def test_partition_counts_match_eda_stats():
    sizes = eda_stats()["sizes"]
    expected = {"train": 94_493, "val": 18_647, "test": 31_825, "ood": 12_584}
    for split, (n, pct) in _partition_table().items():
        assert n == expected[split] == sizes[split]["n"], split
        assert pct == round(sizes[split]["busy_pct"], 1), f"{split}: {pct} vs {sizes[split]['busy_pct']}"


def test_counts_in_text_match_eda_stats():
    sizes = eda_stats()["sizes"]
    for split in ("train", "val", "test", "ood"):
        n = sizes[split]["n"]
        assert f"{n:,}" in TEXT, f"{split}: {n:,} no aparece en el README"
    line = next(ln for ln in LINES if "La verificación compara" in ln)
    nxt = LINES[LINES.index(line) + 1] + " " + LINES[LINES.index(line) + 2]
    snippet = line + " " + nxt
    for split in ("train", "val", "test"):
        m = re.search(rf"{split} ([\d,]+)", snippet)
        assert m and _int(m.group(1)) == sizes[split]["n"], split
    m = re.search(r"OOD ([\d,]+)", snippet)
    assert m and _int(m.group(1)) == sizes["ood"]["n"]


def test_dataset_table_totals():
    sizes = eda_stats()["sizes"]
    ext_total = sizes["train"]["n"] + sizes["val"]["n"] + sizes["test"]["n"]
    m = re.search(r"^\| CNR-EXT \(patches 150×150\) \|[^|]*\| ([\d,]+) \|", TEXT, flags=re.M)
    assert m and _int(m.group(1)) == ext_total == 144_965
    m = re.search(r"^\| CNRPark A/B \|[^|]*\| ([\d,]+) \|", TEXT, flags=re.M)
    assert m and _int(m.group(1)) == sizes["ood"]["n"]


def test_counts_match_download_script():
    """Los conteos que el README dice que se verifican son los de scripts/download_data.py."""
    tree = ast.parse((REPO / "scripts" / "download_data.py").read_text(encoding="utf-8"))
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    exp = consts["EXPECTED"]
    start = TEXT.index("La verificación compara")
    snippet = TEXT[start:start + 400]
    assert f"train {exp['train']:,}" in snippet
    assert f"val {exp['val']:,}" in snippet
    assert f"test {exp['test']:,}" in snippet
    assert f"OOD {exp['ood']:,}" in snippet
    assert f"{exp['csv_rows']:,} filas del CSV" in snippet
    assert f"{exp['ext_unique_slots']} espacios únicos" in snippet
    assert f"{exp['full_frames']:,} frames" in snippet
    assert f"{exp['camera_csvs']} CSV de cámaras" in snippet
    assert consts["SENTINEL"] in TEXT
    for split, pct in consts["EXPECTED_BUSY_PCT"].items():
        assert _partition_table()[split][1] == pct, split


def test_eda_claims_match_stats():
    st = eda_stats()
    assert st["val_rainy"] == 0  # "La validación oficial no tiene días de lluvia"
    assert st["date_overlap_total"] == 0  # particiones separadas por fecha
    m = re.search(r"en (\d+) patches \(([\d.]+) %\)", TEXT)
    assert m, "no se encontró la frase de diferencias con el CSV"
    assert int(m.group(1)) == st["label_mismatch"]
    assert float(m.group(2)) == round(st["label_mismatch_pct"], 3)


# ---------------------------------------------------------------------------
# Afirmaciones sobre configuración
# ---------------------------------------------------------------------------

def test_python_version_and_marker_match_pyproject():
    py = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["project"]["requires-python"] == ">=3.11"
    assert "Python 3.11 o superior" in TEXT
    markers = py["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("data:") for m in markers)


def test_performance_claims_match_configs():
    def cfg(name):
        return yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))

    assert cfg("b2_malexnet")["data"]["batch_size"] == 256
    assert cfg("k3_mnv3_ft")["data"]["batch_size"] == 128
    assert cfg("k5_efficientnet_b0")["data"]["batch_size"] == 64
    for name in ("b2_malexnet", "k1_malexnet_aug", "k2_mnv3_head", "k3_mnv3_ft",
                 "k4_mnv3_ft_cost", "k5_efficientnet_b0", "k6_mnv3_ft_seeds"):
        c = cfg(name)
        assert c["data"]["num_workers"] == 6, name
        assert c["select"]["scenario"] == "comercial", name
        assert c["train"].get("cudnn_benchmark", False) is False, name
    k3, k6 = cfg("k3_mnv3_ft"), cfg("k6_mnv3_ft_seeds")
    assert k6["name"] == "k6_mnv3_ft_seeds"
    for sec in ("model", "data", "optim", "loss"):
        assert k6[sec] == k3[sec], f"K6 difiere de K3 en {sec}"
    assert cfg("k4_mnv3_ft_cost")["loss"]["cost_weight_busy"] == 3


def test_value_config_claims():
    com = yaml.safe_load((REPO / "configs" / "value" / "comercial.yaml").read_text(encoding="utf-8"))
    t_base = com["params"]["t_base_min"]
    assert {"value", "source", "confidence"} <= set(t_base)
    assert t_base["value"] == 6.0 and t_base["confidence"] == "literatura"
    assert com["params"]["precision_verificacion"]["value"] == 0.95
    costs = yaml.safe_load((REPO / "configs" / "costs.yaml").read_text(encoding="utf-8"))
    assert "costo_por_1000_pred_Q" in costs and "cpu_server_usd_per_hour" in costs


# ---------------------------------------------------------------------------
# Enlaces internos
# ---------------------------------------------------------------------------

def _slug(heading: str) -> str:
    """Ancla al estilo de GitHub: minúsculas, sin puntuación, espacios → '-'."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def test_internal_anchors_resolve():
    in_fence = False
    anchors = set()
    for line in LINES:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        m = re.match(r"^#{1,6} (.+)$", line)
        if m and not in_fence:
            anchors.add(_slug(m.group(1)))
    links = set(re.findall(r"\]\(#([^)]+)\)", TEXT))
    broken = sorted(links - anchors)
    assert not broken, f"enlaces internos rotos: {broken}"
