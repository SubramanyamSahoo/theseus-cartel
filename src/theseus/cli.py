from __future__ import annotations

import argparse
import asyncio
import copy
import json
import signal
from pathlib import Path

from .common import load_config, project_root


def smoke_config(cfg):
    c = copy.deepcopy(cfg)
    c["inference"]["backend"] = "mock"
    c["execution"]["results_root"] = "validation/mock"
    c["protocol"].update(
        development_seeds=1,
        test_seeds=1,
        founder_round_candidates=[3, 5],
        founder_rounds=3,
        positive_control_rounds=3,
        turnover_gap=2,
        post_turnover_rounds=4,
        terminal_window=2,
        establishment_window=2,
        test_positive_control_rounds=3,
        deviation_probe={
            "after_full_turnover_rounds": 1,
            "baseline_window": 1,
            "response_window": 1,
            "recovery_window": 1,
            "action_rule": "best_response_to_last_rival",
        },
    )
    c["statistics"]["bootstrap_mc_se"] = 0.05
    scenarios = [
        {
            "id": f"mock_{d}_{j}",
            "dataset": f"mock_{d}",
            "product": str(j),
            "description": "Synthetic validation fixture; not real research data",
            "split": "development" if j == 0 else "test",
            "price_scale": 1.0 + j,
            "volume_scale": 100.0,
            "normalized_costs": [1.0, 1.0],
        }
        for d in range(2)
        for j in range(3)
    ]
    return c, scenarios


async def guarded_run(coro):
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    return await coro


def main():
    parser = argparse.ArgumentParser(
        description="Reproducible model-turnover study on the Apart pricing harness"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["fetch-upstream", "prepare-data", "doctor", "run", "smoke", "workload"]:
        item = sub.add_parser(name)
        item.add_argument("--config", type=Path, default=project_root() / "configs/h100.yaml")
        if name in ["run", "smoke"]:
            item.add_argument("--resume", type=Path)
        if name == "smoke":
            item.add_argument("--scripted", action="store_true")
    item = sub.add_parser("analyze")
    item.add_argument("run_directory", type=Path)
    item = sub.add_parser("suite")
    item.add_argument("--neutral-config", type=Path, default=project_root() / "configs/h100.yaml")
    item.add_argument("--planted-config", type=Path, default=project_root() / "configs/planted.yaml")
    item.add_argument("--resume-suite", type=Path)
    for name in ("campaign", "campaign-smoke", "campaign-plan"):
        item = sub.add_parser(name)
        item.add_argument("--config", type=Path, default=project_root() / "configs/campaign.yaml")
        if name == "campaign":
            item.add_argument("--resume", type=Path)
    item = sub.add_parser("analyze-campaign")
    item.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "analyze-campaign":
        from .campaign_analysis import analyze_campaign

        analyze_campaign(args.directory)
        return
    if args.command in {"campaign", "campaign-smoke", "campaign-plan"}:
        from .campaign import load_campaign, run_campaign

        spec, base = load_campaign(args.config)
        if args.command == "campaign-plan":
            p, d, i = base["protocol"], base["data"], base["inference"]
            roots = 2 * d["test_products_per_dataset"] * len(p["directions"])
            development = (
                2 * d["development_products_per_dataset"] * len(p["directions"]) * p["development_seeds"]
            )
            cells = len(spec["pairs"]) * len(spec["regimes"])
            tuning = len(base["models"]) * (
                1
                + i["autotune_repeats"]
                * sum(min(x, i["max_concurrency_per_model"]) for x in i["autotune_candidates"])
            )
            calls = {
                h: cells
                * (
                    2
                    * (
                        development * (max(p["founder_round_candidates"]) + p["positive_control_rounds"])
                        + roots
                        * (
                            h
                            + p["test_positive_control_rounds"]
                            + len(p["arms"]) * (p["turnover_gap"] + p["post_turnover_rounds"])
                        )
                    )
                    + tuning
                )
                for h in p["founder_round_candidates"]
            }
            print(
                json.dumps(
                    {
                        "models": spec["models"],
                        "pairs": spec["pairs"],
                        "directed_transfers": 2 * len(spec["pairs"]),
                        "blocks_per_seed_wave": cells,
                        "branches_per_seed_wave": cells * roots * len(p["arms"]),
                        "uncached_calls_per_seed_wave_by_founder_horizon": calls,
                        "maximum_waves": spec["target_seeds"],
                        "budget_hours": spec["budget_hours"],
                        "hard_stop_utc": spec.get("hard_stop_utc"),
                        "runtime": "Not predicted from v1 alone; after one full wave, the next starts only if its measured duration plus headroom fits. No idle padding.",
                    },
                    indent=2,
                )
            )
        elif args.command == "campaign-smoke":
            base, scenarios = smoke_config(base)
            base["inference"]["mock_policy"] = "history_copy"
            spec.update(target_seeds=1, hard_stop_utc=None)
            import uuid

            destination = project_root() / "validation/campaign" / uuid.uuid4().hex[:8]
            asyncio.run(
                guarded_run(run_campaign(spec, base, prepared_scenarios=scenarios, destination=destination))
            )
        else:
            asyncio.run(guarded_run(run_campaign(spec, base, args.resume)))
        return
    if args.command == "suite":
        from .suite import run_suite

        configs = [load_config(args.neutral_config), load_config(args.planted_config)]
        asyncio.run(guarded_run(run_suite(configs, args.resume_suite)))
        return
    if args.command == "analyze":
        from .analysis import analyze

        analyze(args.run_directory)
        return
    cfg = load_config(args.config)
    if args.command == "fetch-upstream":
        from .upstream import fetch_upstream

        print(json.dumps(fetch_upstream(cfg), indent=2))
    elif args.command == "prepare-data":
        from .data import prepare

        print(prepare(cfg))
    elif args.command == "doctor":
        from .hardware import capacity_plan
        from .upstream import verify_upstream

        print(json.dumps({"upstream": verify_upstream(cfg), "capacity": capacity_plan(cfg)}, indent=2))
    elif args.command == "workload":
        p, d, i = cfg["protocol"], cfg["data"], cfg["inference"]
        roots = 2 * d["test_products_per_dataset"] * len(p["directions"]) * p["test_seeds"]
        dev = 2 * d["development_products_per_dataset"] * len(p["directions"]) * p["development_seeds"]
        tuning = len(cfg["models"]) * (
            1
            + i["autotune_repeats"]
            * sum(min(x, i["max_concurrency_per_model"]) for x in i["autotune_candidates"])
        )
        calls = {
            h: 2
            * (
                dev * (max(p["founder_round_candidates"]) + p["positive_control_rounds"])
                + roots
                * (
                    h
                    + p["test_positive_control_rounds"]
                    + len(p["arms"]) * (p["turnover_gap"] + p["post_turnover_rounds"])
                )
            )
            + tuning
            for h in p["founder_round_candidates"]
        }
        print(
            json.dumps(
                {
                    "experiment": cfg["experiment"],
                    "test_founders": roots,
                    "test_branches": roots * len(p["arms"]),
                    "test_positive_controls": roots,
                    "uncached_completions_before_transport_retries": calls,
                },
                indent=2,
            )
        )
    else:
        from .runner import pipeline

        if args.command == "smoke":
            cfg, scenarios = smoke_config(cfg)
            if args.scripted:
                cfg["experiment"] = "planted"
                cfg["protocol"]["founder_prefix"] = "PC"
                cfg["inference"]["mock_policy"] = "history_copy"
                cfg["execution"]["results_root"] = "validation/scripted"
            asyncio.run(guarded_run(pipeline(cfg, args.resume, prepared_scenarios=scenarios)))
        else:
            asyncio.run(guarded_run(pipeline(cfg, args.resume)))


if __name__ == "__main__":
    main()
