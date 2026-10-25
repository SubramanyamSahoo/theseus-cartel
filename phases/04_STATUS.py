"""Report an explicitly identified campaign, without treating exit zero as success."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", type=Path)
    group.add_argument("--log", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run
    if run is None:
        log = args.log
        pointer = root / "logs/latest_campaign_log.txt"
        if log is None and pointer.exists():
            log = Path(pointer.read_text().strip())
        if log is None or not log.exists():
            print("No real campaign log is recorded yet. Inspect the full setup log.")
            return
        paths = [
            line.split("=", 1)[1]
            for line in log.read_text().splitlines()
            if line.startswith("CAMPAIGN_DIRECTORY=")
        ]
        if len(paths) != 1:
            print(f"No unique campaign manifest was identified. Inspect {log}")
            return
        run = Path(paths[0])
    path = run / "campaign.json"
    if not path.exists():
        raise SystemExit(f"Missing campaign manifest: {path}")
    record = json.loads(path.read_text())
    print(f"Campaign: {run.resolve()}\nDeadline UTC: {record['deadline_utc']}")
    print("Recorded stop reason:", record.get("stop_reason", "not recorded yet"))
    print("Recorded complete waves:", record.get("complete_waves", []))
    for block in record["blocks"]:
        print(f"{block['status']:12} {block['id']}")
        if block.get("error"):
            print("  Error:", block["error"])
    print("Analysis:", run / "analysis")
    print("A manifest is an execution record. Inspect the analysis and controls for scientific findings.")


if __name__ == "__main__":
    main()
