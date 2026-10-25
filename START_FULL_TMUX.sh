#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
for tool in tmux git python3 nvidia-smi flock; do
  if ! command -v "$tool" >/dev/null; then
    printf 'Missing required command: %s\n' "$tool"
    echo 'On a Lambda GPU instance, install basic tools with: sudo apt-get update && sudo apt-get install -y git tmux unzip python3-venv util-linux'
    exit 1
  fi
done
session="theseus-full-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p logs
log="$PWD/logs/$session.log"
printf -v launch '%q ' bash "$PWD/phases/RUN_ALL.sh" "$log" "$@"
tmux new-session -d -s "$session" -c "$PWD" "$launch"
printf '%s\n' "$session" > logs/latest_full_session.txt
printf 'Started full phased run: %s\nAttach: tmux attach -t %s\nLog: %s\n' "$session" "$session" "$log"
