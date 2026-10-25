#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
echo 'PHASE 1: GPU check, environment, upstream source, tests, and real-data preparation'
nvidia-smi
df -h "$PWD"
bash BOOTSTRAP.sh
