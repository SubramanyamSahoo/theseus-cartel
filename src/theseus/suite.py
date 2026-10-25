from __future__ import annotations

import asyncio
import fcntl
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .common import atomic_json, load_json, project_root, utcnow
from .runner import new_run, pipeline, source_fingerprint


async def run_suite(configs, resume=None):
    if [c["experiment"] for c in configs] != ["neutral", "planted"]:
        raise ValueError("Suite requires neutral followed by planted configurations")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = (
        Path(resume).resolve()
        if resume
        else project_root() / "results/suites" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"configs": configs, "source_hash": source_fingerprint()}
    path = directory / "suite.json"
    with (directory / ".suite.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = (
            load_json(path)
            if path.exists()
            else {"identity": identity, "created_utc": utcnow(), "studies": []}
        )
        if record["identity"] != identity:
            raise ValueError("Suite code/configuration changed; create a new suite")
        print(f"SUITE_DIRECTORY={directory}", flush=True)
        failed = []
        for index, cfg in enumerate(configs):
            if len(record["studies"]) <= index:
                run = new_run(cfg)
                record["studies"].append(
                    {"experiment": cfg["experiment"], "run_directory": str(run), "status": "pending"}
                )
            entry = record["studies"][index]
            if entry["status"] == "complete":
                continue
            entry.update(status="running", updated_utc=utcnow())
            atomic_json(path, record)
            try:
                await pipeline(cfg, entry["run_directory"])
                entry.update(status="complete", updated_utc=utcnow())
                entry.pop("error", None)
            except asyncio.CancelledError:
                entry.update(status="interrupted", updated_utc=utcnow())
                atomic_json(path, record)
                raise
            except Exception as exc:
                entry.update(status="failed", error=str(exc), updated_utc=utcnow())
                failed.append(cfg["experiment"])
                print(
                    f"{cfg['experiment']} failed: {exc}. Continuing the separately labelled study.",
                    flush=True,
                )
            atomic_json(path, record)
        record["finished_utc"] = utcnow()
        atomic_json(path, record)
        lines = [
            "# Two-study suite",
            "",
            "Neutral and PC-instructed-founder results are separate. Do not pool their estimates.",
            "",
        ]
        for entry in record["studies"]:
            lines += [
                f"- {entry['experiment']}: {entry['status']} - {entry['run_directory']}/analysis/report.md"
            ]
        (directory / "summary.md").write_text("\n".join(lines) + "\n")
        if failed:
            raise RuntimeError("Suite finished with failed studies: " + ", ".join(failed))
    return directory
