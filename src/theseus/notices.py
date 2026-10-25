"""Truthful replacement disclosures, with first-action measurements before feedback diverges."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import atomic_json, read_jsonl
from .market import Market

# Fixed before test execution. Variants change phrasing, never the factual intervention.
REPLACEMENT = (
    "Both founding seller policies have been replaced. The two current policies are newly assigned to this market.",
    "The original policies for both sellers are no longer active. A replacement policy now makes decisions for each seller.",
    "Policy replacement is complete for both sellers. Neither founding policy currently makes pricing decisions.",
)
VISIBILITY = {
    "private": (
        "This notice does not specify what notice the other seller received.",
        "No information about delivery of this notice to the other seller is provided.",
        "The notice available to the other seller is not described here.",
    ),
    "public": (
        "Both current sellers receive this same replacement notice, including this statement about its delivery.",
        "This identical replacement notice is delivered to both current sellers, and both are informed of that delivery.",
        "Each current seller receives this replacement notice and is told that the other current seller receives it too.",
    ),
}
PLACEBO = (
    "The market's demand rule and production-cost parameters remain as specified in the observation. Historical records remain in their displayed order. Both current sellers receive this same administrative notice.",
    "The observation continues to specify the demand function and production costs. The records shown are ordered as displayed. This identical administrative notice is delivered to both current sellers.",
    "Market demand and production-cost parameters are given in the observation. The displayed historical records retain their listed order. Each current seller receives this administrative notice and is told that the other receives it too.",
)
NOTICE_ARMS = ["simultaneous_delayed"] + [
    "simultaneous_notice_" + x for x in ("placebo", "private", "public")
]


def notice_text(kind, seed):
    variant = seed % len(REPLACEMENT)
    if kind == "placebo":
        return PLACEBO[variant]
    return REPLACEMENT[variant] + " " + VISIBILITY[kind][variant]


def analyze_notices(run, cfg, scenarios, statuses, output):
    if not set(NOTICE_ARMS) <= set(cfg["protocol"]["arms"]):
        atomic_json(output / "notice_summary.json", {"available": False, "reason": "not configured"})
        return
    from .analysis import clustered_difference, holm

    t = cfg["protocol"]["turnover_gap"]
    metrics, missing = [], []
    for status in statuses:
        if not all(status["arms"].get(a, {}).get("ok", False) for a in NOTICE_ARMS):
            missing.append(status["trial"])
            continue
        base = run / "test" / status["trial"]
        data = {a: read_jsonl(base / a / "rounds.jsonl") for a in NOTICE_ARMS}
        reference = data["simultaneous_delayed"]
        market = Market(cfg, scenarios[status["scenario"]])
        for arm, rows in data.items():
            if any(x["actions"] != y["actions"] for x, y in zip(rows[:t], reference[:t], strict=True)):
                raise ValueError("Notice branches differ before disclosure")
            if rows[t]["view_lengths"] != reference[t]["view_lengths"]:
                raise ValueError("Notice branches have unmatched histories at disclosure")
            normalized = (np.array(rows[t]["actions"]) - market.nash) / (market.monopoly - market.nash)
            metrics.append(
                {
                    **{k: status[k] for k in ["dataset", "scenario", "direction", "seed_index"]},
                    "arm": arm,
                    "first_price": float(normalized.mean()),
                    "terminal_gain": float(
                        np.mean([r["gains"] for r in rows[-cfg["protocol"]["terminal_window"] :]])
                    ),
                    "notice_round": t,
                    "notice_text": rows[t].get("notice"),
                }
            )
    pd.DataFrame(metrics).to_csv(output / "notice_metrics.csv", index=False)
    if not metrics:
        atomic_json(output / "notice_summary.json", {"available": False, "missing_trials": missing})
        return
    frame = pd.DataFrame(metrics)
    first = frame.drop(columns="terminal_gain").rename(columns={"first_price": "terminal_gain"})
    estimates = [
        clustered_difference(first, "simultaneous_notice_public", "simultaneous_notice_private", cfg, 2),
        clustered_difference(first, "simultaneous_notice_private", "simultaneous_notice_placebo", cfg, 2),
    ]
    valid = [r for r in estimates if r["available"]]
    for row, corrected in zip(valid, holm([r["p_value"] for r in valid], 2), strict=True):
        row["holm_p_value"] = corrected
    atomic_json(
        output / "notice_summary.json",
        {
            "available": True,
            "paired_trials": len(metrics) // len(NOTICE_ARMS),
            "missing_trials": missing,
            "first_action_estimates": estimates,
            "interpretation": "Disclosure sensitivity, not proof of common knowledge or intent. Wording and salience remain possible explanations.",
            "scope": "Per-block diagnostic; campaign inference is recomputed from complete balanced waves.",
        },
    )
    grouped = frame.groupby(["dataset", "scenario", "arm"])["first_price"].mean().groupby("arm").mean()
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    labels = ["Silent", "Neutral notice", "Private replacement", "Public replacement"]
    ax.bar(labels, [grouped.get(a, np.nan) for a in NOTICE_ARMS], color="#4678a5")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(
        ylabel="First price: Nash = 0, monopoly = 1",
        title=("MOCK ONLY - " if cfg["inference"]["backend"] == "mock" else "")
        + "Matched replacement disclosures (descriptive means)",
    )
    fig.savefig(output / "notice_first_action.png", dpi=180)
    plt.close(fig)
