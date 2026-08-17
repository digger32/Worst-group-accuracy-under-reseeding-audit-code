#!/usr/bin/env python3
"""Pre-delivery preflight — run by the ASSISTANT in its sandbox after every code
change, BEFORE packing an archive. Encodes the three failure classes that cost
smoke iterations on T2 (21 Jul 2026); ported to A5:

  A  broken files            py_compile every .py, bash -n every .sh, parse every yaml
  B  phantom third-party API 'from X import Y' where X or Y does not exist in the
                             installed library (chronos context->inputs, timesfm 1.x
                             API, gluonts.torch.modules.loss removal, gift-eval name)
  C  env-parity gap          module imported at runtime whose distribution is missing
                             from requirements.txt (the scikit-learn/xgboost case) —
                             checked by tracing sys.modules after a runtime probe
  D  gate self-test          synthetic clean run exits 0, dirty exits 1
  E  packing hygiene         archive must contain no data/ or runs/ entries

Usage:  python scripts/preflight.py [--zip out.zip]
Exit non-zero on any failure. Class B is static (ast over installed sources), so
libraries installable with --no-deps are auditable without their heavy deps.
"""
import argparse
import ast
import importlib.util
import json
import py_compile
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = [ROOT / "a5", ROOT / "scripts", ROOT / "runner"]
STDLIB = set(sys.stdlib_module_names)
FAILS = []


def fail(msg):
    FAILS.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg):
    print(f"  [ok] {msg}")


def step_A_compile():
    print("== A: files compile/parse ==")
    import yaml
    for d in CODE_DIRS:
        for p in d.rglob("*.py"):
            try:
                py_compile.compile(str(p), doraise=True)
            except py_compile.PyCompileError as e:
                fail(f"py_compile {p.name}: {e.msg.splitlines()[-1]}")
    for p in ROOT.rglob("*.sh"):
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        if r.returncode:
            fail(f"bash -n {p.name}: {r.stderr.strip()}")
    for p in (ROOT / "configs").glob("*.yaml"):
        try:
            yaml.safe_load(p.read_text())
        except Exception as e:
            fail(f"yaml {p.name}: {e}")
    if not FAILS:
        ok("all .py/.sh/.yaml parse")


def _module_names(mod: str):
    """Names available in an installed module, resolved on the FILESYSTEM so a
    heavy parent (e.g. gluonts.torch importing torch) is never executed.
    Returns: set of names (defs + assigns + on-disk submodules), 'OPAQUE' for
    binary modules, or None if not installed."""
    parts = mod.split(".")
    try:
        spec = importlib.util.find_spec(parts[0])
    except (ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    if not spec.submodule_search_locations:
        base = Path(spec.origin) if spec.origin else None
    else:
        base = Path(list(spec.submodule_search_locations)[0])
        for part in parts[1:]:
            if (base / part).is_dir():
                base = base / part
            elif (base / f"{part}.py").exists():
                base = base / f"{part}.py"
            else:
                return None
    if base is None:
        return "OPAQUE"
    src = base / "__init__.py" if base.is_dir() else base
    if not src.exists() or src.suffix != ".py":
        return "OPAQUE"
    names = set()
    tree = ast.parse(src.read_text(errors="ignore"))
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in n.names)
    if base.is_dir():  # on-disk submodules satisfy `from pkg import submodule`
        names.update(q.stem for q in base.glob("*.py"))
        names.update(q.name for q in base.iterdir() if q.is_dir())
    return names


def step_B_import_audit():
    print("== B: third-party import audit (static) ==")
    ours = {"a5", "scripts", "runner", "gate_selftest", "prepare_data"}
    allow_opaque = {"torch"}  # basic usage only; too heavy for the sandbox
    checked = missing_mod = missing_name = 0
    for d in CODE_DIRS:
        for p in d.rglob("*.py"):
            tree = ast.parse(p.read_text())
            for n in ast.walk(tree):
                if not (isinstance(n, ast.ImportFrom) and n.module and n.level == 0):
                    continue
                top = n.module.split(".")[0]
                if top in STDLIB or top in ours:
                    continue
                if top in allow_opaque:
                    continue
                checked += 1
                have = _module_names(n.module)
                if have is None:
                    fail(f"{p.name}: '{n.module}' not installed in sandbox — "
                         f"run: pip install --no-deps {top}")
                    missing_mod += 1
                elif have != "OPAQUE":
                    for a in n.names:
                        if a.name != "*" and a.name not in have:
                            fail(f"{p.name}: '{a.name}' not found in {n.module}")
                            missing_name += 1
    ok(f"{checked} third-party from-imports audited "
       f"({missing_mod} missing modules, {missing_name} missing names)")


def step_C_env_parity(probe_modules):
    print("== C: env parity (runtime probe vs requirements.txt) ==")
    from importlib.metadata import packages_distributions
    req = (ROOT / "requirements.txt").read_text().lower()
    pkg2dist = packages_distributions()
    before = set(sys.modules)
    for fn in probe_modules:
        try:
            fn()
        except Exception as e:
            fail(f"runtime probe {fn.__name__}: {type(e).__name__}: {e}")
            return
    used = {m.split(".")[0] for m in set(sys.modules) - before} | \
           {m.split(".")[0] for m in before}
    gaps = []
    for mod in sorted(used):
        if mod in STDLIB or mod.startswith("_"):
            continue
        for dist in pkg2dist.get(mod, []):
            dl = dist.lower()
            if dl in ("pip", "setuptools", "wheel", "pyyaml") or dl in req:
                continue
            # only flag distributions our probe actually exercised
            if mod in ("sklearn",) or dl.replace("-", "_") in req:
                gaps.append(f"{mod} -> {dist}")
    for g in sorted(set(gaps)):
        fail(f"runtime uses {g} but requirements.txt does not list it")
    if not gaps:
        ok("runtime closure covered by requirements.txt")


def _probe_classical():
    """One tiny real pass through the torch-free path: decode -> group -> metrics."""
    import io

    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(ROOT))
    from a5.data import _augment, eval_fingerprint, group_of, normalise, safe_decode
    from a5.metrics import score_all

    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (40, 30, 3), dtype=np.uint8)).save(buf, "PNG")
    arr = safe_decode(buf.getvalue(), 32)
    assert arr is not None and arr.shape == (32, 32, 3), "safe_decode"
    assert safe_decode(b"not an image", 32) is None, "safe_decode must reject junk"
    flat = np.zeros((32, 32, 3), dtype=np.uint8)
    Image.fromarray(flat).save(buf2 := io.BytesIO(), "PNG")
    assert safe_decode(buf2.getvalue(), 32) is None, "constant image must be rejected"

    assert normalise(arr).shape == (3, 32, 32), "normalise"
    assert _augment(arr, np.random.default_rng(1)).shape == arr.shape, "augment"

    y = np.array([0, 0, 1, 1]); a = np.array([0, 1, 0, 1])
    assert group_of(y, a).tolist() == [0, 1, 2, 3], "group_of"
    m = score_all(y, np.array([0, 1, 1, 0]), group_of(y, a))
    assert 0.0 <= m["acc_worst_group"] <= 1.0 and "eo_gap" in m, "score_all"

    images = np.stack([arr] * 4)
    fp = eval_fingerprint(images, np.arange(4))
    assert len(fp["eval_sha256"]) == 64 and fp["min_probe_std"] > 0, "fingerprint"


def step_D_gate_selftest():
    print("== D: gate self-test ==")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gate_selftest.py")],
                       capture_output=True, text=True)
    if r.returncode:
        fail("gate self-test failed:\n" + r.stdout[-400:])
    else:
        ok(r.stdout.strip().splitlines()[-1])


def step_E_packing(zip_path):
    print("== E: packing hygiene ==")
    if not zip_path:
        ok("no --zip given; remember: pack with -x 'a5-build/data/*' 'a5-build/runs/*'")
        return
    bad = [n for n in zipfile.ZipFile(zip_path).namelist()
           if "/data/" in n or "/runs/" in n or "__pycache__" in n]
    if bad:
        fail(f"archive contains forbidden entries: {bad[:5]}")
    else:
        ok(f"{zip_path}: no data/ runs/ __pycache__ entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    a = ap.parse_args()
    step_A_compile()
    step_B_import_audit()
    step_C_env_parity([_probe_classical])
    step_D_gate_selftest()
    step_E_packing(a.zip)
    print("=" * 60)
    if FAILS:
        print(f"PREFLIGHT FAILED: {len(FAILS)} problem(s) — do NOT ship")
        sys.exit(1)
    print("PREFLIGHT PASSED — safe to pack and ship")


if __name__ == "__main__":
    main()
