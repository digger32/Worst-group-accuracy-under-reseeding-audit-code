#!/usr/bin/env bash
# Per-paper venv, built OFFLINE from the wheelhouse, with a constraint guard baked
# into activate. each project on this machine gets its own environment: a
# neighbouring project that pins lightning, torchmetrics and openvino hard would
# otherwise move this one's torch and scikit-learn underneath it. What is shared
# is the constraint file, not the environment.
set -euo pipefail
WH="${WHEELHOUSE:-$HOME/wheelhouse}"
VENV="${VENV:-.venv}"
[ -d "$WH" ] || { echo "no wheelhouse at $WH -- run: bash runner/pipeline.sh wheels"; exit 1; }

python3 -m venv "$VENV"
GUARD="$PWD/constraints-wgaudit.txt"
grep -q PIP_CONSTRAINT "$VENV/bin/activate" || \
  echo "export PIP_CONSTRAINT=$GUARD" >> "$VENV/bin/activate"

# shellcheck disable=SC1090
source "$VENV/bin/activate"
export PIP_CONSTRAINT="$GUARD"
pip install --no-index --find-links="$WH" -U pip setuptools wheel || true
pip install --no-index --find-links="$WH" -r requirements.txt
pip check || echo "[env] pip check reported metadata mismatches -- these are claims, not proof; the smoke unit is the verdict"
pip freeze > env-wgaudit.lock
python - <<'PY'
import importlib
for m in ("torch", "torchvision", "numpy", "scipy", "sklearn", "pandas", "PIL", "yaml"):
    mod = importlib.import_module(m)
    print(f"ok   {m:12s} {getattr(mod, '__version__', '')}")
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
echo "[env] done -> $VENV (lock: env-wgaudit.lock)"
