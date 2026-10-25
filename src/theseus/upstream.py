from __future__ import annotations

import importlib.util
import subprocess
from functools import lru_cache
from pathlib import Path

from .common import atomic_json, file_hash, project_root


def upstream_path(cfg: dict) -> Path:
    p = Path(cfg["upstream"]["directory"])
    return p if p.is_absolute() else project_root() / p


def verify_upstream(cfg: dict) -> dict:
    p = upstream_path(cfg)
    got = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=p, text=True).strip()
    if got != cfg["upstream"]["commit"]:
        raise RuntimeError(f"Upstream commit mismatch: {got}")
    rel = "llm-cartel/environment/oligopoly.py"
    expected = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=p)
    if (p / rel).read_bytes() != expected:
        raise RuntimeError("Upstream market source was modified; use the pinned clean checkout")
    return {"url": cfg["upstream"]["url"], "commit": got, "market_sha256": file_hash(p / rel)}


def fetch_upstream(cfg: dict) -> dict:
    p = upstream_path(cfg)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", cfg["upstream"]["url"], str(p)],
            check=True,
        )
        subprocess.run(["git", "checkout", "--detach", cfg["upstream"]["commit"]], cwd=p, check=True)
    result = verify_upstream(cfg)
    atomic_json(project_root() / "UPSTREAM.lock.json", result)
    return result


def market_class(cfg: dict):
    p = upstream_path(cfg) / "llm-cartel/environment/oligopoly.py"
    return _load_market(str(p))


@lru_cache(maxsize=4)
def _load_market(path: str):
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(
            "Missing supplied codebase. Run: theseus fetch-upstream --config configs/h100.yaml"
        )
    spec = importlib.util.spec_from_file_location("apart_pinned_oligopoly", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OligopolyGame
