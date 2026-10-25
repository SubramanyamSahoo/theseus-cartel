"""Aggregate only complete, predeclared seed waves; retain incomplete cells visibly."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import clustered_difference, holm
from .common import atomic_json, load_json


def analyze_campaign(directory):
    from .campaign import complete_waves

    directory = Path(directory)
    record = load_json(directory / "campaign.json")
    output = directory / "analysis"
    output.mkdir(exist_ok=True)
    spec, cfg = record["identity"]["spec"], record["identity"]["base"]
    complete = complete_waves(record)
    branch_tables, notice_tables, control_tables, coverage = [], [], [], []
    pair_tables = []
    for entry in record["blocks"]:
        path = Path(entry["run_directory"])
        # Manifest paths are relative to the original VM when an archive is downloaded.
        if not path.exists():
            path = directory / "studies" / entry["id"]
        eligible = entry["wave"] in complete and entry["status"] == "complete"
        quality = load_json(path / "COMPLETE.json") if (path / "COMPLETE.json").exists() else {}
        coverage.append(
            {
                **{k: entry[k] for k in ["id", "wave", "pair", "regime", "status"]},
                "included_in_primary": eligible,
                "elapsed_seconds": entry.get("elapsed_seconds"),
                "failed_arms": quality.get("failed_arms"),
                "planned_arms": quality.get("planned_arms"),
                "failed_test_positive_controls": quality.get("failed_test_positive_controls"),
                "error": entry.get("error"),
            }
        )
        for filename, collection in [
            ("episode_metrics.csv", branch_tables),
            ("notice_metrics.csv", notice_tables),
            ("positive_controls.csv", control_tables),
        ]:
            file = path / "analysis" / filename
            if not file.exists() or file.stat().st_size < 3:
                continue
            frame = pd.read_csv(file).assign(
                pair=entry["pair"], regime=entry["regime"], wave=entry["wave"], included_in_primary=eligible
            )
            collection.append(frame)
        ef = path / "analysis/effects.csv"
        if ef.exists():
            pair_tables.append(
                pd.read_csv(ef).assign(pair=entry["pair"], regime=entry["regime"], wave=entry["wave"])
            )
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    branches = pd.concat(branch_tables, ignore_index=True) if branch_tables else pd.DataFrame()
    notices = pd.concat(notice_tables, ignore_index=True) if notice_tables else pd.DataFrame()
    controls = pd.concat(control_tables, ignore_index=True) if control_tables else pd.DataFrame()
    for name, frame in [
        ("all_branch_metrics", branches),
        ("all_notice_metrics", notices),
        ("all_test_controls", controls),
    ]:
        frame.to_csv(output / (name + ".csv"), index=False)
    if pair_tables:
        pd.concat(pair_tables, ignore_index=True).to_csv(
            output / "per_block_effects_exploratory.csv", index=False
        )
    estimates = []
    # Three hypotheses per founder regime; six planned tests across the campaign.
    for regime in spec["regimes"]:
        definitions = [
            ("history_reset", "sequential_full", "sequential_joint_reset", "terminal_gain", branches),
            (
                "public_vs_private_first_price",
                "simultaneous_notice_public",
                "simultaneous_notice_private",
                "first_price",
                notices,
            ),
            (
                "public_vs_placebo_persistence",
                "simultaneous_notice_public",
                "simultaneous_notice_placebo",
                "terminal_gain",
                notices,
            ),
        ]
        for name, left, right, metric, frame in definitions:
            if frame.empty:
                estimates.append({"regime": regime, "hypothesis": name, "available": False})
                continue
            sub = frame[(frame.regime == regime) & frame.included_in_primary].copy()
            if sub.empty:
                estimates.append({"regime": regime, "hypothesis": name, "available": False})
                continue
            if metric != "terminal_gain":
                sub = sub.drop(columns="terminal_gain").rename(columns={metric: "terminal_gain"})
            result = clustered_difference(sub, left, right, cfg, 6)
            estimates.append({"regime": regime, "hypothesis": name, "endpoint": metric, **result})
    valid = [r for r in estimates if r["available"]]
    for row, p in zip(valid, holm([r["p_value"] for r in valid], 6), strict=True):
        row["holm_p_value"] = p
    atomic_json(output / "primary_effects.json", estimates)
    pd.DataFrame(estimates).to_csv(output / "primary_effects.csv", index=False)
    atomic_json(
        output / "summary.json",
        {
            "complete_balanced_waves": complete,
            "target_waves": spec["target_seeds"],
            "completed_blocks": sum(e["status"] == "complete" for e in record["blocks"]),
            "primary_family_size": 6,
            "backend": cfg["inference"]["backend"],
            "source_hash": record["identity"]["source_hash"],
            "cluster_gate_passed": len(valid) == 6
            and all(
                r.get("clusters", 0) >= cfg["statistics"]["minimum_clusters_for_confirmatory_label"]
                for r in valid
            ),
            "scope": "Fixed models, contexts, simulator, and protocol. Product clustering does not turn shared normalized games into independent economies.",
        },
    )
    marker = (
        "MOCK / SCRIPTED VALIDATION ONLY"
        if cfg["inference"]["backend"] == "mock"
        else "RECORDED MODEL EXPERIMENTS - REVIEW BEFORE CLAIMS"
    )
    lines = [
        "# The Cartel After Its Members",
        "",
        marker,
        "",
        f"Complete balanced seed waves: {len(complete)}/{spec['target_seeds']}. Blocks in a partial wave are retained in the CSVs and excluded from the pooled primary estimates.",
        "",
        "The six primary comparisons are fixed in advance. Pairing includes model pair, direction, context, and seed. Repetitions and fixed model pairs are averaged within product context; uncertainty resamples products within dataset. These are context-conditioned estimates, not a random sample of model families or independent real markets.",
        "",
        "| Regime | Hypothesis | Effect | Adjusted interval | Holm p | Contexts |",
        "|---|---|---:|---|---:|---:|",
    ]
    for r in estimates:
        if r["available"]:
            lines.append(
                f"| {r['regime']} | {r['hypothesis']} | {r['effect']:.5f} | [{r['ci_low']:.5f}, {r['ci_high']:.5f}] | {r['holm_p_value']:.4g} | {r['clusters']} |"
            )
        else:
            lines.append(f"| {r['regime']} | {r['hypothesis']} | unavailable | | | |")
    if not controls.empty:
        c = controls[controls.included_in_primary]
        lines += [
            "",
            f"Test PC controls in included waves: {int(c.ok.sum())}/{len(c)} executed successfully. Their observed gains are in all_test_controls.csv; completion is not proof that PC elicited high profits.",
        ]
    lines += [
        "",
        "## Interpretation",
        "",
        "History effects test dependence on inherited observations. First-action disclosure effects are measured before the new history diverges. Public versus private wording is a test of disclosure sensitivity, not an identified measurement of common knowledge, beliefs, or intent. The administrative placebo and fixed paraphrases reduce some prompt confounds without eliminating them.",
        "",
        "Each block also reports the matched-time turnover contrast, deviation probe, and three nonstrategic impulse-response baselines. Similarity to or difference from these limited baselines does not prove or refute strategic collusion. Inspect both-seller gain, unilateral incentives, consumer surplus, null outcomes, and raw failures.",
        "",
        "## Limits and dual use",
        "",
        "No training or scratchpad was added. Models use their native chat templates and a common price-only output grammar. Cross-family contrasts conflate checkpoint architecture, training, and template differences. Shared numeric anchors remain possible. Retail inputs supply descriptive contexts and scales, not identified demand or cartel labels. PC formation and neutral formation remain separate. Replacing agents while keeping observations could fail as a remedy; no real-world cartel capability is established.",
        "",
        "The original v1 run is a separate exploratory baseline. It is not merged into this campaign. This document and child PDFs are review drafts; use the official template for submission.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    if valid:
        fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        for j, r in enumerate(valid):
            ax.plot([r["ci_low"], r["ci_high"]], [j, j], color="#245780")
            ax.scatter(r["effect"], j, color="#245780")
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_yticks(
            range(len(valid)),
            [r["regime"] + ": " + r["hypothesis"].replace("_", " ") for r in valid],
            fontsize=8,
        )
        ax.set(xlabel="Paired normalized effect (endpoint differs by row)", title=marker)
        fig.savefig(output / "primary_effects.png", dpi=180)
        plt.close(fig)
    if not branches.empty and complete:
        frame = branches[branches.included_in_primary]
        transfer = (
            frame[frame.arm.isin(["sequential_full", "sequential_joint_reset"])]
            .groupby(["regime", "pair", "direction", "arm"])
            .terminal_gain.mean()
            .unstack("arm")
            .reindex(columns=["sequential_full", "sequential_joint_reset"])
        )
        transfer["history_minus_reset"] = transfer["sequential_full"] - transfer["sequential_joint_reset"]
        transfer.to_csv(output / "transfer_matrix.csv")
        labels = [f"{x[0]} | {x[1]} | {x[2]}" for x in transfer.index]
        values = transfer[["sequential_full", "sequential_joint_reset", "history_minus_reset"]].to_numpy()
        fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.28)), constrained_layout=True)
        finite = abs(values[np.isfinite(values)])
        extent = max(1, finite.max()) if finite.size else 1
        plot = ax.imshow(
            values,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-extent,
            vmax=extent,
        )
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xticks(range(3), ["Retained history", "Joint reset", "Difference"], fontsize=8)
        ax.set_title(marker + "\nDescriptive transfer matrix")
        fig.colorbar(plot, ax=ax, label="Normalized profit gain")
        fig.savefig(output / "transfer_matrix.png", dpi=180)
        plt.close(fig)
    from .campaign_report import render_campaign_report

    render_campaign_report(directory)
    return output
