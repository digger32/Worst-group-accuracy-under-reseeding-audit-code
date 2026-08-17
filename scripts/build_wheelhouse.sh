#!/usr/bin/env bash
# Build the wheelhouse ONCE, with network. Every venv afterwards is built offline
# from it, which is what makes per-paper environments cheap enough to be separate.
#
# The pip HTTP cache alone is NOT installable offline: `--no-index` needs an index,
# and ~/.cache/pip is not one. A wheelhouse is a directory of built wheels, and
# `--find-links` treats it as a local index. That is the difference.
set -euo pipefail
WH="${WHEELHOUSE:-$HOME/wheelhouse}"
mkdir -p "$WH"
python3 -m pip download --dest "$WH" -r requirements.txt
python3 -m pip wheel --wheel-dir "$WH" --find-links "$WH" -r requirements.txt || true
echo "[wheels] $(ls "$WH" | wc -l) files in $WH"
echo "[wheels] from now on: pip install --no-index --find-links=$WH -r requirements.txt"
