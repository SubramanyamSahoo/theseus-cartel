#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/theseus ]]; then
  echo 'Run phase 1 first.'
  exit 1
fi
echo 'PHASE 3: real model campaign; checkpoint weights download as required'
exec .venv/bin/theseus campaign "$@"
