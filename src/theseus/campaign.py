"""A time-bounded, balanced multi-model campaign on one GPU."""

from __future__ import annotations

import asyncio
import copy
import fcntl
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from .common import atomic_json, load_config, load_json, project_root, utcnow, validate_config
from .runner import pipeline, source_fingerprint


def load_campaign(path):
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text())
    if spec.get("campaign_schema") != 1:
        raise ValueError("Unknown campaign schema")
    if spec["budget_hours"] <= 0 or spec["target_seeds"] < 1 or spec["wave_headroom_fraction"] < 0:
        raise ValueError("Invalid campaign budget")
    if set(spec["regimes"]) != {"neutral", "planted"} or len(spec["regimes"]) != 2:
        raise ValueError("Both separate founder regimes are required")
    pairs = [tuple(p) for p in spec["pairs"]]
    if len(set(pairs)) != len(pairs) or any(len(p) != 2 or p[0] == p[1] for p in pairs):
        raise ValueError("Unique two-model pairs required")
    base = load_config(path.parent / spec["base_config"])
    for pair in pairs:
        for regime in spec["regimes"]:
            make_config(base, spec, pair, regime, 0)
    if spec.get("hard_stop_utc") and datetime.fromisoformat(spec["hard_stop_utc"]).tzinfo is None:
        raise ValueError("Hard stop must have a timezone")
    return spec, base


def make_config(base, spec, pair, regime, wave):
    cfg = copy.deepcopy(base)
    cfg["experiment"] = regime
    cfg["campaign_context"] = {"pair": list(pair), "wave": wave, "regime": regime}
    cfg["protocol"].update(
        founder_prefix="PC" if regime == "planted" else "P0", test_seeds=1, test_seed_start=wave
    )
    for role, key in zip(("founder", "replacement"), pair, strict=True):
        model = spec["models"][key]
        cfg["models"][role].update(id=model["id"], revision=model["revision"], family=model["family"])
    return validate_config(cfg)


def wave_fits(seconds_left, durations, headroom):
    return not durations or seconds_left >= max(durations) * (1 + headroom)


def complete_waves(record):
    count = len(record["identity"]["spec"]["pairs"]) * len(record["identity"]["spec"]["regimes"])
    return [
        w
        for w in range(record["identity"]["spec"]["target_seeds"])
        if len([e for e in record["blocks"] if e["wave"] == w and e["status"] == "complete"]) == count
    ]


async def run_campaign(spec, base, resume=None, *, prepared_scenarios=None, destination=None):
    if resume:
        directory = Path(resume).resolve()
        if not (directory / "campaign.json").exists():
            raise ValueError("Campaign resume path has no manifest")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = (
            Path(destination)
            if destination
            else project_root() / "results/campaigns" / f"{stamp}_{uuid.uuid4().hex[:8]}"
        )
        directory.mkdir(parents=True, exist_ok=False)
    identity = {"spec": spec, "base": base, "source_hash": source_fingerprint()}
    path = directory / "campaign.json"
    with (directory / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            record = load_json(path)
            if record["identity"] != identity:
                raise ValueError("Campaign code/configuration changed; create a new campaign")
        else:
            now = datetime.now(timezone.utc)
            deadline = now + timedelta(hours=spec["budget_hours"])
            if spec.get("hard_stop_utc"):
                deadline = min(deadline, datetime.fromisoformat(spec["hard_stop_utc"]))
            record = {
                "identity": identity,
                "created_utc": now.isoformat(),
                "deadline_utc": deadline.isoformat(),
                "blocks": [],
                "wave_seconds": {},
            }
        print(f"CAMPAIGN_DIRECTORY={directory}\nGLOBAL_DEADLINE_UTC={record['deadline_utc']}", flush=True)
        deadline = datetime.fromisoformat(record["deadline_utc"])
        atomic_json(path, record)
        reason = "target_seeds_completed"
        try:
            for wave in range(spec["target_seeds"]):
                if wave in complete_waves(record):
                    continue
                remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
                begun = any(e["wave"] == wave for e in record["blocks"])
                if remaining <= 0 or (
                    not begun
                    and not wave_fits(
                        remaining, list(record["wave_seconds"].values()), spec["wave_headroom_fraction"]
                    )
                ):
                    reason = "time_budget_no_complete_wave_expected"
                    break
                for pair in spec["pairs"]:
                    pair_name = "__".join(pair)
                    for regime in spec["regimes"]:
                        name = f"seed{wave}__{pair_name}__{regime}"
                        matches = [e for e in record["blocks"] if e["id"] == name]
                        if matches:
                            entry = matches[0]
                        else:
                            run = directory / "studies" / name
                            run.mkdir(parents=True)
                            entry = {
                                "id": name,
                                "wave": wave,
                                "pair": pair_name,
                                "regime": regime,
                                "run_directory": str(run),
                                "status": "pending",
                                "elapsed_seconds": 0.0,
                            }
                            record["blocks"].append(entry)
                        if entry["status"] == "complete":
                            continue
                        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
                        if remaining <= 0:
                            raise TimeoutError("Campaign wall-time deadline reached")
                        cfg = make_config(base, spec, pair, regime, wave)
                        entry.update(status="running", started_utc=utcnow())
                        atomic_json(path, record)
                        started = time.perf_counter()
                        try:
                            async with asyncio.timeout(remaining):
                                await pipeline(
                                    cfg, entry["run_directory"], prepared_scenarios=prepared_scenarios
                                )
                            entry.update(status="complete", completed_utc=utcnow())
                            entry.pop("error", None)
                        except (asyncio.CancelledError, TimeoutError) as exc:
                            entry.update(status="interrupted", error=type(exc).__name__)
                            raise
                        except Exception as exc:
                            entry.update(status="failed", error=str(exc))
                            print(f"Block {name} failed: {exc}", flush=True)
                        finally:
                            entry["elapsed_seconds"] += time.perf_counter() - started
                            atomic_json(path, record)
                if wave not in complete_waves(record):
                    reason = "incomplete_wave_due_to_failed_blocks"
                    break
                record["wave_seconds"][str(wave)] = sum(
                    e["elapsed_seconds"] for e in record["blocks"] if e["wave"] == wave
                )
                atomic_json(path, record)
                from .campaign_analysis import analyze_campaign

                analyze_campaign(directory)
        except TimeoutError:
            reason = "time_budget_reached_partial_wave_preserved"
        except asyncio.CancelledError:
            reason = "user_interrupted"
            raise
        finally:
            record.update(stop_reason=reason, updated_utc=utcnow(), complete_waves=complete_waves(record))
            atomic_json(path, record)
            from .campaign_analysis import analyze_campaign

            analyze_campaign(directory)
            print(f"Campaign stopped: {reason}. Results: {directory}/analysis", flush=True)
            print("GPU jobs are stopped. The VM remains billed until you terminate it.", flush=True)
    return directory
