"""Nonstrategic response baselines starting at each observed pre-probe checkpoint."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import atomic_json, read_jsonl
from .market import Market

POLICIES = ("copy_rival", "hold_own", "myopic_best_response")


def response_path(market, previous, policy, rounds, deviator, force):
    previous = np.array(previous, float)
    result = []
    for t in range(rounds):
        if policy == "copy_rival":
            prices = previous[::-1].copy()
        elif policy == "hold_own":
            prices = previous.copy()
        elif policy == "myopic_best_response":
            prices = np.array([market.best_response(a, previous[1 - a]) for a in range(2)])
        else:
            raise ValueError("Unknown null policy")
        if force and t == 0:
            prices[deviator] = market.best_response(deviator, previous[1 - deviator])
        result.append(prices.copy())
        previous = prices
    return np.array(result)


def analyze_nulls(run, cfg, scenarios, statuses, output):
    rows = []
    t = cfg["protocol"]["turnover_gap"] + cfg["protocol"]["deviation_probe"]["after_full_turnover_rounds"]
    for status in statuses:
        if not status["arms"].get("sequential_full", {}).get("ok", False):
            continue
        observed = read_jsonl(run / "test" / status["trial"] / "sequential_full/rounds.jsonl")
        market = Market(cfg, scenarios[status["scenario"]])
        first = observed[0]["events"][0]["agent"]
        deviator, responder = 1 - first, first
        previous = observed[t - 1]["actions"]
        for policy in POLICIES:
            a = response_path(market, previous, policy, len(observed) - t, deviator, True)
            b = response_path(market, previous, policy, len(observed) - t, deviator, False)
            delta = (a[:, responder] - b[:, responder]) / (
                market.monopoly[responder] - market.nash[responder]
            )
            for k, value in enumerate(delta):
                rows.append(
                    {
                        **{f: status[f] for f in ["dataset", "scenario", "direction", "seed_index"]},
                        "policy": policy,
                        "relative_round": k,
                        "response_delta": float(value),
                    }
                )
    pd.DataFrame(rows).to_csv(output / "null_response_curves.csv", index=False)
    atomic_json(
        output / "null_baselines.json",
        {
            "policies": list(POLICIES),
            "llm_calls": 0,
            "initialization": "Each recorded sequential_full branch immediately before its matched probe round",
            "scope": "Reference countermodels, not fitted mechanisms or ground-truth collusion labels. Similarity does not establish equivalence.",
        },
    )
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    frame = pd.DataFrame(rows)
    for policy in POLICIES:
        curve = (
            frame[frame.policy == policy]
            .groupby(["dataset", "scenario", "relative_round"])
            .response_delta.mean()
            .groupby("relative_round")
            .mean()
        )
        ax.plot(curve.index, curve.values, label=policy.replace("_", " "), linestyle="--")
    actual_path = output / "deviation_round_metrics.csv"
    if actual_path.exists():
        actual = pd.read_csv(actual_path)
        actual = (
            actual[actual.relative_round >= 0]
            .groupby(["dataset", "scenario", "relative_round"])
            .response_delta.mean()
            .groupby("relative_round")
            .mean()
        )
        ax.plot(actual.index, actual.values, label="Observed model response", color="black", linewidth=2)
    ax.set(
        xlabel="Rounds after forced action",
        ylabel="Normalized responder-price difference",
        title=("MOCK ONLY - " if cfg["inference"]["backend"] == "mock" else "")
        + "Observed response and nonstrategic baselines",
    )
    ax.legend(fontsize=8)
    fig.savefig(output / "response_countermodels.png", dpi=180)
    plt.close(fig)
