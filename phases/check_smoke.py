"""Check this invocation's smoke manifest, not only its process exit code."""

import json
import sys
from pathlib import Path


def check(log):
    paths = [
        line.split("=", 1)[1]
        for line in Path(log).read_text().splitlines()
        if line.startswith("CAMPAIGN_DIRECTORY=")
    ]
    if len(paths) != 1:
        raise RuntimeError("Could not identify exactly one smoke campaign in the log")
    directory = Path(paths[0])
    record = json.loads((directory / "campaign.json").read_text())
    spec = record["identity"]["spec"]
    expected = len(spec["pairs"]) * len(spec["regimes"]) * spec["target_seeds"]
    if record["identity"]["base"]["inference"]["backend"] != "mock":
        raise RuntimeError("This is not a synthetic validation run")
    if len(record["blocks"]) != expected or any(b["status"] != "complete" for b in record["blocks"]):
        raise RuntimeError(
            "Synthetic campaign did not complete every planned block; real campaign not started"
        )
    for entry in record["blocks"]:
        complete = json.loads((directory / "studies" / entry["id"] / "COMPLETE.json").read_text())
        if complete["failed_arms"] or complete["failed_test_positive_controls"]:
            raise RuntimeError("Synthetic branch or control failures; real campaign not started")
    print(f"Synthetic validation passed: {expected} completed blocks. This is not GPU evidence.")


if __name__ == "__main__":
    check(sys.argv[1])
