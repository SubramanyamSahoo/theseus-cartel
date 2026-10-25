#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/theseus ]]; then echo 'Run bash BOOTSTRAP.sh first.'; exit 1; fi
session="theseus-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p logs
log="$PWD/logs/$session.log"
printf -v launch '%q ' bash "$PWD/scripts/run_session.sh" "$log" "$@"
tmux new-session -d -s "$session" -c "$PWD" "$launch"
printf 'Started %s\nAttach: tmux attach -t %s\nLog: %s\n' "$session" "$session" "$log"
printf '%s\n' "$session" > logs/latest_session.txt
