from __future__ import annotations

import hashlib
import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts) -> int:
    # vLLM accepts signed 32-bit seeds. Hash namespaces prevent execution-order dependence.
    return int(digest(parts)[:16], 16) % np.iinfo(np.int32).max


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_json(path: Path):
    return json.loads(Path(path).read_text())


async def gather_clean(*coroutines):
    tasks = [asyncio.ensure_future(c) for c in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def collect_completed(coroutines, callback):
    tasks = [asyncio.ensure_future(c) for c in coroutines]
    results = []
    try:
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            callback(result, len(results), len(tasks))
        return results
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = []
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            # A crash may leave one torn final record. Truncate only that record.
            Path(path).write_text("".join(json.dumps(x) + "\n" for x in rows))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        f.flush()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    def read(file, seen):
        file = file.resolve()
        if file in seen:
            raise ValueError("Circular configuration inheritance")
        value = yaml.safe_load(file.read_text())
        base = value.pop("extends", None)
        if base:
            parent = read(file.parent / base, seen | {file})

            def merge(a, b):
                for key, item in b.items():
                    if isinstance(item, dict) and isinstance(a.get(key), dict):
                        merge(a[key], item)
                    else:
                        a[key] = item

            merge(parent, value)
            return parent
        return value

    cfg = read(Path(path), set())
    return validate_config(cfg)


def validate_config(cfg):
    if cfg.get("schema_version") != 3:
        raise ValueError("Unsupported configuration schema")
    for role in ("founder", "replacement"):
        spec = cfg["models"][role]
        if not spec["id"].startswith(("Qwen/", "mistralai/", "allenai/")):
            raise ValueError("Use a reviewed official model organization")
        if len(spec["revision"]) != 40 or any(c not in "0123456789abcdef" for c in spec["revision"]):
            raise ValueError("Pin each model to an immutable commit")
    if cfg["models"]["founder"]["id"] == cfg["models"]["replacement"]["id"]:
        raise ValueError("Real turnover requires different model checkpoints")
    p = cfg["protocol"]
    allowed_arms = {
        "no_turnover",
        "simultaneous_full",
        "simultaneous_delayed",
        "sequential_full",
        "sequential_drop_pre",
        "sequential_joint_reset",
        "sequential_competitive",
        "fresh_final",
        "sequential_shuffle",
        "sequential_deviation",
        "simultaneous_notice_placebo",
        "simultaneous_notice_private",
        "simultaneous_notice_public",
    }
    if not set(p["arms"]) <= allowed_arms or len(set(p["arms"])) != len(p["arms"]):
        raise ValueError("Unknown or duplicated intervention arm")
    if not set(p["directions"]) <= {"forward", "reverse"}:
        raise ValueError("Unknown model turnover direction")
    if cfg["models"]["founder"]["port"] == cfg["models"]["replacement"]["port"]:
        raise ValueError("Models require separate ports")
    if cfg["inference"]["backend"] not in {"vllm", "mock"}:
        raise ValueError("Unknown inference backend")
    if cfg.get("experiment") not in {"neutral", "planted"}:
        raise ValueError("Specify the neutral or planted experiment")
    expected = "PC" if cfg["experiment"] == "planted" else "P0"
    if p["founder_prefix"] != expected or p["newcomer_prefix"] != "P0":
        raise ValueError("Experiment label and founder/successor instructions disagree")
    if cfg["inference"]["mock_policy"] not in {"nash", "history_copy"}:
        raise ValueError("Unknown scripted validation policy")
    if p.get("test_seed_start", 0) < 0:
        raise ValueError("Seed offset cannot be negative")
    probe = p["deviation_probe"]
    if (
        probe["action_rule"] != "best_response_to_last_rival"
        or not 0 < probe["baseline_window"] <= probe["after_full_turnover_rounds"]
        or probe["response_window"] <= 0
        or probe["recovery_window"] <= 0
        or probe["after_full_turnover_rounds"] + 1 + probe["response_window"] + probe["recovery_window"]
        > p["post_turnover_rounds"]
    ):
        raise ValueError("Probe windows must fit after full turnover with separate response/recovery periods")
    if p["test_positive_control_rounds"] < p["terminal_window"]:
        raise ValueError("Positive-control duration must cover the terminal scoring window")
    if cfg["hardware"]["context_headroom_fraction"] < 0:
        raise ValueError("Context headroom cannot be negative")
    if (
        min(
            cfg["market"]["history_window"],
            p["initial_nash_rounds"],
            p["turnover_gap"],
            p["test_seeds"],
            p["development_seeds"],
            cfg["execution"]["max_parallel_episodes"],
            cfg["inference"]["max_concurrency_per_model"],
            cfg["inference"]["autotune_repeats"],
        )
        <= 0
    ):
        raise ValueError("Lengths, repeat counts, and concurrency limits must be positive")
    if not 0 < cfg["statistics"]["alpha"] < 1 or not 0 < cfg["statistics"]["bootstrap_mc_se"] < 0.5:
        raise ValueError("Invalid statistical error tolerance")
    if not 0 < p["terminal_window"] <= p["post_turnover_rounds"]:
        raise ValueError("terminal_window must fit inside the post-turnover period")
    if not 0 < cfg["data"]["temporal_fit_fraction"] < 1:
        raise ValueError("Invalid temporal fit split")
    if not 0 < cfg["hardware"]["combined_memory_fraction"] < 1:
        raise ValueError("GPU memory fraction must be strictly between zero and one")
    if cfg["execution"]["max_cost_usd"] is not None and (
        not cfg["execution"]["hourly_usd"] or not cfg["execution"]["billing_started_utc"]
    ):
        raise ValueError("Cost cutoff requires hourly_usd and billing_started_utc")
    return cfg
