from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import atomic_json, read_jsonl, stable_seed
from .market import Market


def analyze_probes(run, cfg, scenarios, statuses, output, clustered_difference, holm):
    specification = cfg["protocol"].get("deviation_probe")
    if not specification:
        return
    metrics, curves, missing = [], [], []
    for status in statuses:
        if not all(
            status["arms"].get(a, {}).get("ok", False) for a in ["sequential_full", "sequential_deviation"]
        ):
            missing.append(status["trial"])
            continue
        base = run / "test" / status["trial"]
        reference = read_jsonl(base / "sequential_full/rounds.jsonl")
        probe = read_jsonl(base / "sequential_deviation/rounds.jsonl")
        shocks = [r for r in probe if r["intervention"] is not None]
        if len(shocks) != 1:
            raise ValueError("Deviation arm must have exactly one recorded forced action")
        shock = shocks[0]["intervention"]
        t, responder = shock["local_round"], shock["responder"]
        if any(a["actions"] != b["actions"] for a, b in zip(reference[:t], probe[:t], strict=True)):
            raise ValueError("Probe and reference do not share the same pre-intervention trajectory")
        market = Market(cfg, scenarios[status["scenario"]])
        scale = market.monopoly[responder] - market.nash[responder]
        delta = np.array(
            [
                (a["actions"][responder] - b["actions"][responder]) / scale
                for a, b in zip(probe, reference, strict=True)
            ]
        )
        early = float(delta[t + 1 : t + 1 + specification["response_window"]].mean())
        late = float(delta[-specification["recovery_window"] :].mean())
        identity = {k: status[k] for k in ["scenario", "dataset", "direction", "seed_index"]}
        metrics.append(
            {
                **identity,
                "probe_round": t,
                "responder": responder,
                "early_response_delta": early,
                "late_response_delta": late,
                "rebound_delta": late - early,
                "undercuts_last_rival": shock["undercuts_last_rival"],
                "undercuts_current_rival": shock["undercuts_current_rival"],
                "normalized_price_shock": (shock["executed_price"] - shock["proposed_price"])
                / (market.monopoly[shock["agent"]] - market.nash[shock["agent"]]),
                "pre_probe_gain": float(
                    np.mean([r["gains"] for r in probe[t - specification["baseline_window"] : t]])
                ),
                "founder_established": status["founder_established"],
            }
        )
        for k in range(t - specification["baseline_window"], len(probe)):
            curves.append({**identity, "relative_round": k - t, "response_delta": float(delta[k])})
    pd.DataFrame(metrics).to_csv(output / "deviation_metrics.csv", index=False)
    atomic_json(output / "deviation_missing.json", missing)
    if not metrics:
        atomic_json(output / "deviation_summary.json", {"available": False, "missing_trials": len(missing)})
        return
    frame = pd.DataFrame(metrics)
    estimates = []
    for metric in ["early_response_delta", "late_response_delta", "rebound_delta"]:
        contrasted = frame[["dataset", "scenario", "direction", "seed_index", metric]].rename(
            columns={metric: "terminal_gain"}
        )
        zero = contrasted.assign(arm="zero", terminal_gain=0.0)
        contrasted = contrasted.assign(arm=metric)
        estimate = clustered_difference(pd.concat([contrasted, zero]), metric, "zero", cfg, 3)
        estimates.append({"metric": metric, **estimate})
    available = [e for e in estimates if e["available"]]
    for row, corrected in zip(available, holm([e["p_value"] for e in available], 3), strict=True):
        row["holm_p_value"] = corrected
    atomic_json(
        output / "deviation_summary.json",
        {
            "available": True,
            "paired_trials": len(frame),
            "missing_trials": len(missing),
            "actual_undercut_trials": int(frame["undercuts_current_rival"].sum()),
            "estimates": estimates,
            "inference": "Exploratory diagnostic family, separate from primary hypotheses",
            "interpretation": "Temporary response and recovery are compatible with punishment, but reactive anchoring can also generate such patterns; not proof of strategic intent.",
        },
    )
    raw = pd.DataFrame(curves)
    raw.to_csv(output / "deviation_round_metrics.csv", index=False)
    clusters = (
        raw.groupby(["dataset", "scenario", "relative_round"])["response_delta"]
        .mean()
        .unstack("relative_round")
    )
    rng = np.random.default_rng(stable_seed(cfg["seed"], "probe-curve"))
    draws = math.ceil(0.25 / cfg["statistics"]["bootstrap_mc_se"] ** 2)
    bootstrap = np.zeros((draws, len(clusters.columns)))
    for _, group in clusters.groupby(level="dataset"):
        x = group.to_numpy()
        indices = rng.integers(len(x), size=(draws, len(x)))
        bootstrap += x[indices].sum(axis=1) / len(clusters)
    alpha = cfg["statistics"]["alpha"]
    lo, hi = np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2], axis=0)
    plot = pd.DataFrame(
        {
            "relative_round": clusters.columns,
            "effect": clusters.mean().to_numpy(),
            "pointwise_ci_low": lo,
            "pointwise_ci_high": hi,
        }
    )
    plot.to_csv(output / "deviation_plot_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(
        plot["relative_round"], plot["effect"], color="navy", label="Probe minus matched no-probe control"
    )
    ax.fill_between(
        plot["relative_round"],
        lo,
        hi,
        color="steelblue",
        alpha=0.25,
        label="Pointwise product-cluster interval",
    )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="black", linestyle=":", label="One-round forced price")
    title = (
        "SCRIPTED / MOCK DIAGNOSTIC - NOT QWEN EVIDENCE"
        if cfg["inference"]["backend"] == "mock"
        else "Response to a one-round deviation"
    )
    ax.set(
        xlabel="Rounds relative to the forced action",
        ylabel="Responder price difference / (monopoly price - Nash price)",
        title=title,
    )
    ax.legend(fontsize=9)
    fig.savefig(output / "deviation_response.png", dpi=180)
    fig.savefig(output / "deviation_response.pdf")
    plt.close(fig)
