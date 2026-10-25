#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/theseus ]]; then
  echo 'Run phase 1 first.'
  exit 1
fi
mkdir -p logs
echo 'PHASE 2: planned workload, pinned tokenizer checks, and full synthetic campaign'
.venv/bin/theseus campaign-plan | tee logs/campaign_plan.json
.venv/bin/python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
plan = json.loads(Path('logs/campaign_plan.json').read_text())
stop = plan.get('hard_stop_utc')
if stop and datetime.now(timezone.utc) >= datetime.fromisoformat(stop):
    raise SystemExit('The configured campaign deadline has passed. No experiment was started. Review configs/campaign.yaml for a new study; the launcher never silently extends a deadline.')
PY
.venv/bin/python scripts/check_tokenizers.py
smoke_log="$(mktemp "$PWD/logs/smoke.XXXXXX.log")"
.venv/bin/theseus campaign-smoke 2>&1 | tee "$smoke_log"
.venv/bin/python phases/check_smoke.py "$smoke_log"
