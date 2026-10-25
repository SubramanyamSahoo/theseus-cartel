from __future__ import annotations

import asyncio
import copy
import math
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download, snapshot_download

from .common import atomic_json, stable_seed, utcnow
from .inference import action
from .market import Market
from .protocol import initial_history

GIB = 1024**3


def gpu_inventory():
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi is unavailable; use the CPU smoke command for local verification")
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = [x.split(",") for x in output.strip().splitlines()]
    if len(rows) != 1:
        raise RuntimeError("This launcher requires exactly one GPU visible to nvidia-smi")
    name, total, used, driver = [x.strip() for x in rows[0]]
    return {
        "name": name,
        "total_bytes": int(total) * 1024**2,
        "used_bytes": int(used) * 1024**2,
        "driver_version": driver,
    }


def measure_context_tokens(cfg, scenarios, tokenizer_factory=None):
    if tokenizer_factory is None:
        from transformers import AutoTokenizer

        tokenizer_factory = AutoTokenizer.from_pretrained
    development = [s for s in scenarios if s["split"] == "development"]
    if not development:
        raise ValueError("Need development contexts for capacity measurement")
    measured = {}
    for role, spec in cfg["models"].items():
        tokenizer = tokenizer_factory(spec["id"], revision=spec["revision"], trust_remote_code=False)
        lengths = []
        for scenario in development:
            market = Market(cfg, scenario)
            horizon = (
                max(cfg["protocol"]["founder_round_candidates"])
                + cfg["protocol"]["turnover_gap"]
                + cfg["protocol"]["post_turnover_rounds"]
            )
            for prices in (market.nash, market.monopoly):
                history = [
                    dict(market.counterfactual_record({}, prices), t=horizon + k)
                    for k in range(cfg["market"]["history_window"])
                ]
                for prefix in {
                    cfg["protocol"]["founder_prefix"],
                    cfg["protocol"]["newcomer_prefix"],
                    cfg["protocol"]["positive_control_prefix"],
                }:
                    for agent in range(2):
                        ids = tokenizer.apply_chat_template(
                            market.prompt(agent, prefix, history),
                            tokenize=True,
                            add_generation_prompt=True,
                            **(
                                {"enable_thinking": cfg["inference"]["enable_thinking"]}
                                if spec["id"].startswith("Qwen/")
                                else {}
                            ),
                        )
                        lengths.append(len(ids))
                        if any(a.startswith("simultaneous_notice_") for a in cfg["protocol"]["arms"]):
                            from .notices import notice_text, REPLACEMENT

                            for kind in ("placebo", "private", "public"):
                                for variant in range(len(REPLACEMENT)):
                                    prompt = copy.deepcopy(market.prompt(agent, prefix, history))
                                    prompt[-1]["content"] += "\n\nAdministrative notice:\n" + notice_text(
                                        kind, variant
                                    )
                                    extra = (
                                        {"enable_thinking": cfg["inference"]["enable_thinking"]}
                                        if spec["id"].startswith("Qwen/")
                                        else {}
                                    )
                                    lengths.append(
                                        len(
                                            tokenizer.apply_chat_template(
                                                prompt, tokenize=True, add_generation_prompt=True, **extra
                                            )
                                        )
                                    )
        required = max(lengths) + cfg["inference"]["max_output_tokens"]
        if required > cfg["inference"]["max_model_len"]:
            raise ValueError(f"{role}: measured development context exceeds configured context limit")
        measured[role] = {
            "maximum_prompt_tokens": max(lengths),
            "sample_count": len(lengths),
            "reference_context_tokens": min(
                cfg["inference"]["max_model_len"],
                math.ceil(required * (1 + cfg["hardware"]["context_headroom_fraction"])),
            ),
        }
    return measured


def capacity_plan(cfg, measured_contexts=None):
    h, inventory = cfg["hardware"], gpu_inventory()
    if (
        h["expected_gpu"] not in inventory["name"]
        or inventory["total_bytes"] / GIB < h["expected_min_vram_gib"]
    ):
        raise RuntimeError(
            f"Expected a {h['expected_gpu']} with >= {h['expected_min_vram_gib']} GiB: {inventory}"
        )
    total = inventory["total_bytes"]
    # Leave the configured headroom intact; never evict another user's GPU processes.
    if inventory["used_bytes"] > total * (1 - h["combined_memory_fraction"]):
        raise RuntimeError("Too much GPU memory is already occupied. Stop your other GPU jobs first.")
    models = {}
    for role, spec in cfg["models"].items():
        if not spec["revision"] or len(spec["revision"]) != 40:
            raise ValueError("Immutable Hugging Face commit revisions are required")

        def read(name, spec=spec):
            return json.loads(Path(hf_hub_download(spec["id"], name, revision=spec["revision"])).read_text())

        config = read("config.json")
        index = read("model.safetensors.index.json")
        weights = index["metadata"]["total_size"]
        head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        kv_token_bytes = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * head_dim * 2
        models[role] = {
            "weight_bytes": int(weights),
            "weight_files": sorted(set(index.get("weight_map", {}).values())),
            "kv_bytes_per_token": kv_token_bytes,
            "model_config": config,
            "overhead_bytes": int(h["overhead_gib_per_engine"] * GIB),
        }
    remainder = int(total * h["combined_memory_fraction"]) - sum(
        x["weight_bytes"] + x["overhead_bytes"] for x in models.values()
    )
    if remainder <= 0:
        raise RuntimeError("Models plus configured overhead do not fit the memory budget")
    for role, item in models.items():
        # Divide available KV memory equally; size concurrency from actual architecture.
        item["kv_cache_bytes"] = remainder // len(models)
        reference = (
            (measured_contexts or {})
            .get(role, {})
            .get("reference_context_tokens", cfg["inference"]["max_model_len"])
        )
        item["reference_context_tokens"] = reference
        item["maximum_context_tokens"] = cfg["inference"]["max_model_len"]
        context_bytes = item["kv_bytes_per_token"] * reference
        item["max_num_seqs"] = min(h["max_num_seqs_cap"], item["kv_cache_bytes"] // context_bytes)
        if item["max_num_seqs"] < 1:
            raise RuntimeError("No complete configured context fits in the planned KV cache")
        item["gpu_memory_utilization"] = (
            item["weight_bytes"] + item["overhead_bytes"] + item["kv_cache_bytes"]
        ) / total
    return {
        "gpu": inventory,
        "models": models,
        "created_utc": utcnow(),
        "measured_contexts": measured_contexts,
        "scope": "Memory estimate; startup and measured throughput still determine feasibility",
    }


class Servers:
    def __init__(self, cfg, run_dir):
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.processes, self.logs = [], []

    async def start(self, budget, scenarios):
        if self.cfg["inference"]["backend"] == "mock":
            return
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot initialize CUDA. Check the Lambda driver/image.")
        # Exercise CUDA before a multi-minute model load.
        torch.empty(1, device="cuda").add_(1)
        torch.cuda.empty_cache()
        contexts = await asyncio.to_thread(measure_context_tokens, self.cfg, scenarios)
        budget.check()
        plan = capacity_plan(self.cfg, contexts)
        atomic_json(self.run_dir / "hardware.json", plan)
        logdir = self.run_dir / "logs"
        logdir.mkdir(exist_ok=True)
        async with httpx.AsyncClient(trust_env=False) as http:
            for role, spec in self.cfg["models"].items():
                budget.check()
                port = spec["port"]
                # Never silently reuse a stale server with unknown precision/revision.
                import socket

                with socket.socket() as sock:
                    if sock.connect_ex(("127.0.0.1", port)) == 0:
                        raise RuntimeError(f"Port {port} is occupied; stop that server or change the config")
                print(f"Downloading pinned {spec['id']}; then starting on localhost:{port}", flush=True)
                if not plan["models"][role]["weight_files"]:
                    raise ValueError("Pinned safetensors index has no weight shards")
                await asyncio.to_thread(
                    snapshot_download,
                    spec["id"],
                    revision=spec["revision"],
                    allow_patterns=plan["models"][role]["weight_files"]
                    + ["*.json", "*.txt", "*.jinja", "*.tiktoken", "*.model*"],
                )
                budget.check()
                item, h = plan["models"][role], self.cfg["hardware"]
                command = [
                    sys.executable,
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--model",
                    spec["id"],
                    "--revision",
                    spec["revision"],
                    "--tokenizer-revision",
                    spec["revision"],
                    "--served-model-name",
                    spec["id"],
                    "--dtype",
                    "bfloat16",
                    "--max-model-len",
                    str(self.cfg["inference"]["max_model_len"]),
                    "--gpu-memory-utilization",
                    str(item["gpu_memory_utilization"]),
                    "--kv-cache-memory-bytes",
                    str(item["kv_cache_bytes"]),
                    "--max-num-seqs",
                    str(item["max_num_seqs"]),
                    "--max-num-batched-tokens",
                    str(h["max_num_batched_tokens"]),
                    "--enable-chunked-prefill",
                    "--no-enable-log-requests",
                ]
                if h["enable_prefix_caching"]:
                    command.append("--enable-prefix-caching")
                if h["enforce_eager"]:
                    command.append("--enforce-eager")
                env = {
                    **os.environ,
                    "OMP_NUM_THREADS": str(h["cpu_threads_per_engine"]),
                    "TOKENIZERS_PARALLELISM": "false",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                }
                logfile = (logdir / f"vllm_{role}.log").open("a")
                self.logs.append(logfile)
                process = subprocess.Popen(
                    command, stdout=logfile, stderr=subprocess.STDOUT, env=env, start_new_session=True
                )
                self.processes.append(process)
                atomic_json(logdir / f"vllm_{role}_command.json", {"argv": command, "pid": process.pid})
                deadline = time.monotonic() + h["startup_timeout_seconds"]
                while time.monotonic() < deadline:
                    budget.check()
                    if process.poll() is not None:
                        raise RuntimeError(f"{role} server exited: see {logfile.name}")
                    try:
                        response = await http.get(f"http://127.0.0.1:{port}/health", timeout=2)
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(1)
                else:
                    raise TimeoutError(f"{role} startup timed out: see {logfile.name}")

    async def close(self):
        for p in self.processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in self.processes:
            try:
                await asyncio.to_thread(p.wait, timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                await asyncio.to_thread(p.wait)
        for f in self.logs:
            f.close()


async def tune_concurrency(cfg, client, scenarios, output):
    if cfg["inference"]["backend"] == "mock":
        result = {
            "limits": {r: cfg["inference"]["max_concurrency_per_model"] for r in cfg["models"]},
            "measurements": [],
            "backend": "mock",
            "not_a_performance_measurement": True,
        }
        atomic_json(output, result)
        return result["limits"]
    # A full-length development history gives a representative prefill workload.
    scenario = next(s for s in scenarios if s["split"] == "development")
    market = Market(cfg, scenario)
    history = [
        dict(initial_history(cfg, scenario)[0], t=k + 1) for k in range(cfg["market"]["history_window"])
    ]
    measurements = []
    # Warm both engines before timing. Measure contention with both models active.
    for role in cfg["models"]:
        reply = await client.choose(
            role, market.prompt(0, "P0", history), stable_seed(cfg["seed"], "warmup", role), bypass_cache=True
        )
        action(reply)
    for candidate in cfg["inference"]["autotune_candidates"]:
        count = min(candidate, cfg["inference"]["max_concurrency_per_model"])
        rates, errors = [], []
        for rep in range(cfg["inference"]["autotune_repeats"]):
            started = time.perf_counter()
            replies = await asyncio.gather(
                *[
                    client.choose(
                        role,
                        market.prompt(k % 2, "P0", history),
                        stable_seed(cfg["seed"], "throughput", role, count, rep, k),
                        bypass_cache=True,
                    )
                    for role in cfg["models"]
                    for k in range(count)
                ],
                return_exceptions=True,
            )
            for reply in replies:
                try:
                    if isinstance(reply, BaseException):
                        raise reply
                    action(reply)
                except Exception as exc:
                    errors.append(str(exc)[:200])
            rates.append(len(replies) / (time.perf_counter() - started))
        measurements.append(
            {
                "concurrency_per_model": count,
                "rates": rates,
                "requests_per_second": sum(rates) / len(rates),
                "errors": errors,
            }
        )
    valid = [x for x in measurements if not x["errors"]]
    if not valid:
        atomic_json(output, {"measurements": measurements, "limits": {}, "failed": True})
        raise RuntimeError("No batching candidate produced valid outputs; inspect tuning measurements")
    best = max(valid, key=lambda x: (x["requests_per_second"], -x["concurrency_per_model"]))
    limits = {role: best["concurrency_per_model"] for role in cfg["models"]}
    atomic_json(
        output,
        {
            "limits": limits,
            "measurements": measurements,
            "scope": "Best measured symmetric candidate with both engines active on development prompts; not a global optimum",
        },
    )
    return limits
