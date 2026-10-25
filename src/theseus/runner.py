from __future__ import annotations

import asyncio
import copy
import fcntl
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import (
    atomic_json,
    collect_completed,
    digest,
    file_hash,
    gather_clean,
    load_json,
    project_root,
    stable_seed,
    utcnow,
)
from .data import prepare, resolve_path
from .hardware import Servers, tune_concurrency
from .inference import Budget, BudgetExceeded, Client
from .protocol import episode, established, initial_history
from .upstream import fetch_upstream, verify_upstream


def source_fingerprint():
    return digest(
        {
            str(p.relative_to(project_root())): file_hash(p)
            for p in sorted((project_root() / "src").rglob("*.py"))
        }
    )


def new_run(cfg):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = "mock" if cfg["inference"]["backend"] == "mock" else "models"
    root = resolve_path(cfg["execution"]["results_root"])
    directory = root / f"{stamp}_{kind}_{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


async def calibrate(cfg, scenarios, client, run):
    path = run / "calibration.json"
    if path.exists():
        return load_json(path)
    p = cfg["protocol"]
    sem = asyncio.Semaphore(cfg["execution"]["max_parallel_episodes"])

    async def one(scenario, direction, index, positive):
        seed = stable_seed(cfg["seed"], "development", scenario["id"], index)
        label = "positive_control" if positive else "study_founder"
        actual_prefix = p["positive_control_prefix"] if positive else p["founder_prefix"]
        identifier = f"{scenario['id']}__{direction}__{index}__{label}_{actual_prefix}"
        async with sem:
            try:
                rows = await episode(
                    cfg,
                    scenario,
                    client,
                    run / "development" / identifier,
                    direction=direction,
                    seed=seed,
                    arm="founder",
                    rounds=p["positive_control_rounds"] if positive else max(p["founder_round_candidates"]),
                    parent=initial_history(cfg, scenario),
                    prefix=p["positive_control_prefix"] if positive else p["founder_prefix"],
                )
                return {
                    "id": identifier,
                    "direction": direction,
                    "positive": positive,
                    "prefix": actual_prefix,
                    "ok": True,
                    "established_by_horizon": {
                        str(n): established(rows[:n], cfg) for n in p["founder_round_candidates"]
                    }
                    if not positive
                    else {},
                    "terminal_gain": float(np.mean([r["gains"] for r in rows[-p["terminal_window"] :]])),
                }
            except BudgetExceeded:
                raise
            except Exception as exc:
                return {
                    "id": identifier,
                    "direction": direction,
                    "positive": positive,
                    "ok": False,
                    "error": str(exc),
                }

    tasks = [
        one(s, direction, index, positive)
        for s in scenarios
        if s["split"] == "development"
        for direction in p["directions"]
        for index in range(p["development_seeds"])
        for positive in (False, True)
    ]

    def progress(result, done, total):
        print(f"Development {done}/{total}: {result['id']} ok={result['ok']}", flush=True)

    cases = await collect_completed(tasks, progress)
    valid = [x for x in cases if x["ok"] and not x["positive"]]
    if not valid:
        atomic_json(run / "calibration_failed.json", cases)
        raise RuntimeError("No valid study-founder development episodes; fix inference before the test set")
    scores = {
        n: sum(x["established_by_horizon"][str(n)] for x in valid) for n in p["founder_round_candidates"]
    }
    maximum = max(scores.values())
    chosen = min(n for n, score in scores.items() if score == maximum) if maximum else p["founder_rounds"]
    result = {
        "founder_rounds": chosen,
        "scores": scores,
        "cases": cases,
        "selection_rule": "Shortest candidate maximizing established study-founder development histories; configured fallback if all zero",
        "founder_prefix": p["founder_prefix"],
        "experiment": cfg["experiment"],
        "no_founder_formation_in_development": maximum == 0,
        "no_neutral_formation_in_development": maximum == 0 if p["founder_prefix"] == "P0" else None,
        "test_outcomes_used": False,
        "created_utc": utcnow(),
    }
    atomic_json(path, result)
    return result


async def run_test(cfg, scenarios, client, run):
    sem = asyncio.Semaphore(cfg["execution"]["max_parallel_episodes"])
    p = cfg["protocol"]

    async def trial(scenario, direction, index):
        seed = stable_seed(cfg["seed"], "test", scenario["id"], index)
        key = f"{scenario['id']}__{direction}__{index}"
        base = run / "test" / key
        try:
            async with sem:
                await episode(
                    cfg,
                    scenario,
                    client,
                    base / "positive_control",
                    direction=direction,
                    seed=seed,
                    arm="founder",
                    rounds=p["test_positive_control_rounds"],
                    parent=initial_history(cfg, scenario),
                    prefix=p["positive_control_prefix"],
                )
            positive_control = {"ok": True}
        except BudgetExceeded:
            raise
        except Exception as exc:
            positive_control = {"ok": False, "error": str(exc)}
        try:
            async with sem:
                founders = await episode(
                    cfg,
                    scenario,
                    client,
                    base / "founder",
                    direction=direction,
                    seed=seed,
                    arm="founder",
                    rounds=p["founder_rounds"],
                    parent=initial_history(cfg, scenario),
                )
        except BudgetExceeded:
            raise
        except Exception as exc:
            result = {
                "trial": key,
                "scenario": scenario["id"],
                "dataset": scenario["dataset"],
                "direction": direction,
                "seed_index": index,
                "parent_failed": True,
                "positive_control": positive_control,
                "error": str(exc),
                "arms": {a: {"ok": False, "error": "parent failed"} for a in p["arms"]},
            }
            atomic_json(base / "status.json", result)
            return result
        parent = initial_history(cfg, scenario) + founders

        async def branch(arm):
            async with sem:
                try:
                    await episode(
                        cfg,
                        scenario,
                        client,
                        base / arm,
                        direction=direction,
                        seed=seed,
                        arm=arm,
                        rounds=p["turnover_gap"] + p["post_turnover_rounds"],
                        parent=parent,
                    )
                    return arm, {"ok": True}
                except BudgetExceeded:
                    raise
                except Exception as exc:
                    return arm, {"ok": False, "error": str(exc)}

        arms = dict(await gather_clean(*[branch(a) for a in p["arms"]]))
        result = {
            "trial": key,
            "scenario": scenario["id"],
            "dataset": scenario["dataset"],
            "direction": direction,
            "seed_index": index,
            "parent_failed": False,
            "positive_control": positive_control,
            "founder_established": established(founders, cfg),
            "arms": arms,
        }
        atomic_json(base / "status.json", result)
        return result

    tasks = [
        trial(s, d, i)
        for s in scenarios
        if s["split"] == "test"
        for d in p["directions"]
        for i in range(p.get("test_seed_start", 0), p.get("test_seed_start", 0) + p["test_seeds"])
    ]

    def progress(result, done, total):
        atomic_json(
            run / "progress.json",
            {"trials_completed": done, "trials_planned": total, "updated_utc": utcnow()},
        )
        print(f"Test {done}/{total}: {result['trial']}", flush=True)

    return await collect_completed(tasks, progress)


async def pipeline(cfg, resume=None, *, prepared_scenarios=None, external_servers=False):
    run = Path(resume).resolve() if resume else new_run(cfg)
    if not run.is_dir():
        raise ValueError("Resume directory does not exist")
    with (run / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is writing this run directory") from exc
        try:
            return await _pipeline(
                cfg, run, prepared_scenarios=prepared_scenarios, external_servers=external_servers
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


async def _pipeline(cfg, resume=None, *, prepared_scenarios=None, external_servers=False):
    run = Path(resume).resolve() if resume else new_run(cfg)
    if not run.is_dir():
        raise ValueError("Resume directory does not exist")
    print(f"RUN_DIRECTORY={run}", flush=True)
    requested = {"config": cfg, "source_hash": source_fingerprint()}
    if (run / "requested.json").exists():
        if load_json(run / "requested.json") != requested:
            raise ValueError(
                "Code/config changed since this run. Use a new run instead of mixing experiments."
            )
    else:
        atomic_json(run / "requested.json", requested)
    budget = Budget(cfg)
    servers = Servers(cfg, run)
    client = None
    success = False
    try:
        budget.check()
        upstream = fetch_upstream(cfg)
        scenarios = (
            prepared_scenarios
            if prepared_scenarios is not None
            else load_json(await asyncio.to_thread(prepare, cfg))
        )
        if (run / "scenarios.json").exists() and load_json(run / "scenarios.json") != scenarios:
            raise ValueError("Scenario data changed on resume; use a new run")
        atomic_json(run / "scenarios.json", scenarios)
        if prepared_scenarios is None:
            atomic_json(
                run / "data_manifest.json",
                load_json(resolve_path(cfg["data"]["processed_dir"]) / "manifest.json"),
            )
        if not external_servers:
            await servers.start(budget, scenarios)
        elif cfg["inference"]["backend"] != "mock":
            raise ValueError(
                "Externally managed GPU servers are intentionally unsupported: revisions must be auditable"
            )
        client = Client(cfg, run / "response_cache.sqlite", budget)
        await client.verify_models()
        tuning_path = run / "concurrency.json"
        limits = (
            load_json(tuning_path)["limits"]
            if tuning_path.exists()
            else await tune_concurrency(cfg, client, scenarios, tuning_path)
        )
        client.set_limits(limits)
        calibration = await calibrate(cfg, scenarios, client, run)
        effective = copy.deepcopy(cfg)
        effective["protocol"]["founder_rounds"] = calibration["founder_rounds"]
        prereg = {
            "config": effective,
            "upstream": upstream,
            "source_hash": requested["source_hash"],
            "scenario_hash": digest(scenarios),
            "model_revisions": cfg["models"],
            "primary_estimand": "Within-founder paired terminal normalized-profit difference, averaged within product then equally across products",
            "analysis_unit": "dataset/product context; seeds and directions nested within product",
            "all_founder_histories_in_primary_analysis": True,
            "created_before_test_calls": True,
            "backend": cfg["inference"]["backend"],
            "experiment": cfg["experiment"],
            "prompt_disclosure": "Full structural demand, both costs/qualities, explicit two-seller simulation; no scratchpad; syntax-constrained non-thinking decoding",
        }
        frozen = run / "preregistration.json"
        if frozen.exists() and load_json(frozen) != prereg:
            raise ValueError("Preregistration changed on resume")
        atomic_json(frozen, prereg)
        atomic_json(
            run / "environment.json",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "upstream": verify_upstream(cfg),
                "created_utc": utcnow(),
            },
        )
        try:
            packages = subprocess.check_output(
                [str(Path(__import__("sys").executable)), "-m", "pip", "freeze"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            (run / "packages.txt").write_text(packages)
        except subprocess.CalledProcessError:
            import importlib.metadata

            (run / "packages.txt").write_text(
                "\n".join(
                    sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
                )
            )
        statuses = await run_test(effective, scenarios, client, run)
        atomic_json(run / "statuses.json", statuses)
        from .analysis import analyze

        analyze(run)
        failures = sum(not v["ok"] for s in statuses for v in s["arms"].values())
        planned = len(statuses) * len(effective["protocol"]["arms"])
        control_failures = sum(not s["positive_control"]["ok"] for s in statuses)
        limit = cfg["execution"]["max_failed_episode_fraction"]
        success = failures / planned <= limit and control_failures / len(statuses) <= limit
        analysis_quality = (
            load_json(run / "analysis/quality.json") if (run / "analysis/quality.json").exists() else {}
        )
        atomic_json(
            run / "COMPLETE.json",
            {
                "completed_utc": utcnow(),
                "backend": cfg["inference"]["backend"],
                "requests_this_invocation": client.requests,
                "cache_hits_this_invocation": client.hits,
                "failed_arms": failures,
                "planned_arms": planned,
                "planned_test_positive_controls": len(statuses),
                "failed_test_positive_controls": control_failures,
                "failure_rate_gate_passed": success,
                "cluster_gate_passed": analysis_quality.get("cluster_gate_passed", False),
                "quality_gate_passed": success and analysis_quality.get("cluster_gate_passed", False),
            },
        )
        if not success:
            raise RuntimeError("Failure-rate quality gate failed; report generated with missingness warnings")
        return run
    except BaseException as exc:
        atomic_json(
            run / "INTERRUPTED.json", {"time": utcnow(), "type": type(exc).__name__, "error": str(exc)}
        )
        raise
    finally:
        if client is not None:
            await client.close()
        await servers.close()
        if success:
            (run / "INTERRUPTED.json").unlink(missing_ok=True)
        print(
            f"Artifacts: {run}\nGPU server processes stopped. Instance billing continues until you terminate the VM.",
            flush=True,
        )
