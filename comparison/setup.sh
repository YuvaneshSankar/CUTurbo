#!/usr/bin/env bash
# Idempotent setup for the 4-way TurboQuant comparison.
# Creates comparison/.venv, installs cuda-tile + torch (cu128), clones his repo.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip

# cuda-tile: pure-python, ~250 KB
pip install cuda-tile

# torch with CUDA 12.8 runtime — matches driver 570.x line
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

if [ ! -d turboquant_cutile ]; then
  git clone https://github.com/DevTechJr/turboquant_cutile.git
fi

echo "Setup complete. Run:"
echo "  source $HERE/.venv/bin/activate"
echo "  python comparison/check_cutile_env.py   # stage 1 feasibility"
echo "  python comparison/run_4way.py           # stage 3 4-way benchmark"
