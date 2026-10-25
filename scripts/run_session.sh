#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
log="$1"
shift
export PYTHONUNBUFFERED=1
if [[ "${1:-}" == '--campaign' ]]; then
  shift
  theseus_command=(.venv/bin/theseus campaign "$@")
elif [[ "${1:-}" == '--suite' ]]; then
  shift
  theseus_command=(.venv/bin/theseus suite "$@")
else
  theseus_command=(.venv/bin/theseus run --config configs/h100.yaml "$@")
fi
set +e
"${theseus_command[@]}" 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
set -e
printf '\nExperiment exited with status %s. See %s\n' "$status" "$log"
printf 'The Lambda VM remains billed until you terminate it.\n'
exit "$status"
