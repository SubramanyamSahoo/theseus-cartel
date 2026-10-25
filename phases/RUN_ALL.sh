#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
log="${1:?Provide the full-run log path}"
shift
mkdir -p logs
exec > >(tee -a "$log") 2>&1
export PYTHONUNBUFFERED=1
exec 9>logs/.full_phases.lock
if ! flock -n 9; then
  echo 'Another full phased launcher is active in this project.'
  exit 1
fi
if [[ "$#" -eq 0 ]]; then
  bash phases/01_SETUP.sh
  bash phases/02_VALIDATE.sh
elif [[ "$#" -eq 2 && "$1" == '--resume' ]]; then
  echo 'Resuming the recorded campaign directly; setup and smoke are skipped.'
else
  echo 'Usage: bash START_FULL_TMUX.sh [--resume CAMPAIGN_DIRECTORY]'
  exit 2
fi
campaign_log="$(mktemp "$PWD/logs/campaign.XXXXXX.log")"
printf '%s\n' "$campaign_log" > logs/latest_campaign_log.txt
set +e
bash phases/03_RUN.sh "$@" 2>&1 | tee "$campaign_log"
status=${PIPESTATUS[0]}
set -e
python3 phases/04_STATUS.py --log "$campaign_log"
printf '\nCampaign command exited with status %s. The status above shows completed and failed blocks.\n' "$status"
echo 'Lambda billing continues until you terminate the VM. Export and download your results first.'
exit "$status"
