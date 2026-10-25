from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .common import digest


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, cfg):
        self.cfg = cfg["execution"]
        self.started = time.monotonic()

    def check(self):
        e = self.cfg
        if (time.monotonic() - self.started) / 3600 >= e["max_wall_hours"]:
            raise BudgetExceeded("Configured wall-time limit reached; resume the recorded run if desired")
        if e["max_cost_usd"] is not None:
            since = datetime.fromisoformat(e["billing_started_utc"].replace("Z", "+00:00"))
            if since.tzinfo is None:
                raise ValueError("billing_started_utc needs a timezone")
            cost = (datetime.now(timezone.utc) - since).total_seconds() / 3600 * e["hourly_usd"]
            if cost >= e["max_cost_usd"]:
                raise BudgetExceeded("Estimated instance cost reached the configured limit")


def parse_price(text: str) -> float:
    # No repair prompt and no fallback action: either a valid observation or a recorded failure.
    match = re.fullmatch(r"\s*My chosen price:\s*([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*", text)
    if not match:
        raise ValueError(f"Invalid price format: {text[:160]!r}")
    price = float(match.group(1))
    if not math.isfinite(price) or price < 0:
        raise ValueError("Invalid nonfinite/negative price")
    return price


class Client:
    def __init__(self, cfg: dict, cache: Path, budget: Budget | None = None):
        self.cfg, self.budget = cfg, budget or Budget(cfg)
        self.limit = cfg["inference"]["max_concurrency_per_model"]
        self.semaphores = {r: asyncio.Semaphore(self.limit) for r in cfg["models"]}
        self.http = httpx.AsyncClient(
            trust_env=False,
            timeout=cfg["inference"]["request_timeout_seconds"],
            limits=httpx.Limits(max_connections=2 * self.limit, max_keepalive_connections=2 * self.limit),
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.hits, self.requests = 0, 0
        self.key_locks = {}

    def set_limits(self, limits):
        self.semaphores = {r: asyncio.Semaphore(n) for r, n in limits.items()}

    async def close(self):
        await self.http.aclose()
        self.db.close()

    async def verify_models(self):
        if self.cfg["inference"]["backend"] == "mock":
            return
        for role, spec in self.cfg["models"].items():
            response = await self.http.get(f"http://127.0.0.1:{spec['port']}/v1/models")
            response.raise_for_status()
            ids = [x["id"] for x in response.json()["data"]]
            if spec["id"] not in ids:
                raise RuntimeError(f"{role}: incorrect served checkpoint {ids}")

    async def choose(self, role, messages, seed, *, mock_price=None, bypass_cache=False):
        if self.cfg["inference"]["cache_responses"] and not bypass_cache:
            # Coalesce identical in-flight branch requests as well as completed requests.
            # Keep locks for this client lifetime so queued waiters cannot split across locks.
            key = digest(
                (role, messages, seed, mock_price if self.cfg["inference"]["backend"] == "mock" else None)
            )
            lock = self.key_locks.setdefault(key, asyncio.Lock())
            async with lock:
                return await self._choose(role, messages, seed, mock_price=mock_price)
        return await self._choose(role, messages, seed, mock_price=mock_price, bypass_cache=bypass_cache)

    async def _choose(self, role, messages, seed, *, mock_price=None, bypass_cache=False):
        i, spec = self.cfg["inference"], self.cfg["models"][role]
        body = {
            "model": spec["id"],
            "messages": messages,
            "seed": int(seed),
            "temperature": i["temperature"],
            "top_p": i["top_p"],
            "top_k": i["top_k"],
            "max_tokens": i["max_output_tokens"],
        }
        if spec["id"].startswith("Qwen/"):
            body["chat_template_kwargs"] = {"enable_thinking": i["enable_thinking"]}
        if i["structured_price_output"]:
            # Constrain syntax, never the selected numerical action or its magnitude.
            body["structured_outputs"] = {"regex": r"My chosen price: [0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?"}
        key = digest(
            {
                "body": body,
                "revision": spec["revision"],
                "backend": i["backend"],
                "mock_price": mock_price if i["backend"] == "mock" else None,
            }
        )
        self.budget.check()
        if i["cache_responses"] and not bypass_cache:
            cached = self.db.execute("SELECT value FROM responses WHERE key=?", (key,)).fetchone()
            if cached:
                self.hits += 1
                return json.loads(cached[0])
        async with self.semaphores[role]:
            self.budget.check()
            started = time.perf_counter()
            if i["backend"] == "mock":
                if mock_price is None:
                    raise ValueError("Mock price must be supplied explicitly")
                await asyncio.sleep(0)
                payload = {
                    "model": spec["id"],
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": f"My chosen price: {mock_price:.12g}"},
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }
            else:
                url = f"http://127.0.0.1:{spec['port']}/v1/chat/completions"
                for attempt in range(i["transport_retries"] + 1):
                    try:
                        response = await self.http.post(url, json=body)
                        if response.status_code >= 500:
                            response.raise_for_status()
                        if response.is_error:
                            raise ValueError(
                                f"Inference request rejected ({response.status_code}): {response.text[:300]}"
                            )
                        payload = response.json()
                        break
                    except (httpx.TransportError, httpx.HTTPStatusError):
                        if attempt == i["transport_retries"]:
                            raise
                        self.budget.check()
                        await asyncio.sleep(i["transport_backoff_seconds"] * 2**attempt)
            if payload["model"] != spec["id"]:
                raise RuntimeError("Returned model does not match the requested checkpoint")
            choice = payload["choices"][0]
            result = {
                "text": choice["message"]["content"],
                "finish_reason": choice["finish_reason"],
                "model": payload["model"],
                "revision": spec["revision"],
                "usage": payload.get("usage", {}),
                "latency_seconds": time.perf_counter() - started,
                "request_hash": key,
                "seed": seed,
                "backend": i["backend"],
            }
            # Persist invalid outputs too. Resuming must not resample until a preferred action appears.
            self.requests += 1
            if i["cache_responses"] and not bypass_cache:
                self.db.execute("INSERT OR REPLACE INTO responses VALUES (?,?)", (key, json.dumps(result)))
                self.db.commit()
            return result


def action(result):
    if result["finish_reason"] != "stop":
        raise ValueError(f"Incomplete generation: {result['finish_reason']}")
    return parse_price(result["text"])
