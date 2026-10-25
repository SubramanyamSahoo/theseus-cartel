#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! command -v git >/dev/null || ! command -v tmux >/dev/null; then
  echo 'Install git and tmux first (Ubuntu: sudo apt-get update && sudo apt-get install -y git tmux).'
  exit 1
fi
if command -v uv >/dev/null && [[ "$(uv --version)" == "uv 0.13.4"* ]]; then
  theseus_uv="$(command -v uv)"
else
  python3 -m venv .bootstrap-venv
  .bootstrap-venv/bin/python -m pip install 'uv==0.13.4'
  theseus_uv="$PWD/.bootstrap-venv/bin/uv"
fi
"$theseus_uv" python install 3.12
"$theseus_uv" sync --locked --python 3.12 --extra gpu --extra dev
.venv/bin/theseus fetch-upstream --config configs/h100.yaml
.venv/bin/pytest -q
.venv/bin/theseus prepare-data --config configs/h100.yaml
echo 'Setup complete. Print workload: .venv/bin/theseus campaign-plan'
echo 'Start the expanded study: bash START_CAMPAIGN_TMUX.sh'
