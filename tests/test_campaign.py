import asyncio
import copy
import json
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pandas as pd
import pytest

from theseus.analysis import clustered_difference
from theseus.campaign import complete_waves, load_campaign, make_config, wave_fits
from theseus.cli import smoke_config
from theseus.common import atomic_json, load_json, project_root
from theseus.inference import Client
from theseus.market import Market
from theseus.notices import NOTICE_ARMS, analyze_notices, notice_text
from theseus.nulls import response_path
from theseus.protocol import episode


def test_campaign_registry_offsets_and_budget_rules():
    spec, base = load_campaign(project_root() / "configs/campaign.yaml")
    assert len(spec["models"]) == 4
    assert len({m["family"] for m in spec["models"].values()}) == 3
    first = make_config(base, spec, spec["pairs"][1], "planted", 0)
    second = make_config(base, spec, spec["pairs"][1], "neutral", 1)
    assert first["protocol"]["test_seed_start"] == 0
    assert second["protocol"]["test_seed_start"] == 1
    assert first["models"] == second["models"]
    assert first["data"] == second["data"] == base["data"]
    assert first["protocol"]["founder_prefix"] == "PC"
    assert second["protocol"]["founder_prefix"] == "P0"
    assert wave_fits(120, [100, 80], 0.2)
    assert not wave_fits(119, [100, 80], 0.2)


@pytest.mark.parametrize("policy", ["copy_rival", "hold_own", "myopic_best_response"])
def test_nonstrategic_countermodels_have_computable_responses(cfg, scenario, policy):
    market = Market(cfg, scenario)
    previous = market.monopoly
    shock = response_path(market, previous, policy, 5, 1, True)
    control = response_path(market, previous, policy, 5, 1, False)
    assert shock[0, 0] == control[0, 0]  # Simultaneous responder cannot see the shock yet.
    assert shock[0, 1] == market.best_response(1, previous[0])
    if policy == "copy_rival":
        assert shock[1, 0] < control[1, 0]
        assert shock[2, 0] == control[2, 0]  # Copying alone can produce apparent recovery.
    else:
        np.testing.assert_allclose(shock[:, 0], control[:, 0])


def test_non_qwen_request_uses_native_template_without_qwen_switch(cfg, tmp_path):
    cfg = copy.deepcopy(cfg)
    cfg["models"]["founder"]["id"] = "allenai/OLMo-2-1124-7B-Instruct"

    async def check():
        client = Client(cfg, tmp_path / "cache.sqlite")
        await client.http.aclose()

        def handler(request):
            body = json.loads(request.content)
            assert "chat_template_kwargs" not in body
            assert "regex" in body["structured_outputs"]
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "choices": [{"message": {"content": "My chosen price: 1.5"}, "finish_reason": "stop"}],
                },
            )

        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
        try:
            await client.choose("founder", [{"role": "user", "content": "fixture"}], 7)
        finally:
            await client.close()

    asyncio.run(check())


def test_notice_first_action_is_matched_and_copying_is_not_disclosure_sensitive(tmp_path):
    _, cfg = load_campaign(project_root() / "configs/campaign.yaml")
    cfg, scenarios = smoke_config(cfg)
    cfg["inference"]["mock_policy"] = "history_copy"
    scenario = next(s for s in scenarios if s["split"] == "test")
    market = Market(cfg, scenario)
    parent = [{"t": i + 1, **market.evaluate(market.monopoly)} for i in range(3)]
    t = cfg["protocol"]["turnover_gap"]

    async def check():
        client = Client(cfg, tmp_path / "cache.sqlite")
        try:
            results = {}
            for arm in NOTICE_ARMS:
                results[arm] = await episode(
                    cfg,
                    scenario,
                    client,
                    tmp_path / "test/trial" / arm,
                    direction="forward",
                    seed=2,
                    rounds=6,
                    parent=parent,
                    arm=arm,
                )
            for arm, rows in results.items():
                assert all(row["notice"] is None for row in rows[:t])
                assert rows[t]["roles"] == ["replacement", "replacement"]
                assert rows[t]["view_lengths"] == results["simultaneous_delayed"][t]["view_lengths"]
                if arm != "simultaneous_delayed":
                    assert rows[t]["notice"] == notice_text(arm.rsplit("_", 1)[1], 2)
            return results
        finally:
            await client.close()

    asyncio.run(check())
    output = tmp_path / "analysis"
    output.mkdir()
    status = {
        "trial": "trial",
        "scenario": scenario["id"],
        "dataset": scenario["dataset"],
        "direction": "forward",
        "seed_index": 0,
        "arms": {a: {"ok": True} for a in NOTICE_ARMS},
    }
    analyze_notices(tmp_path, cfg, {scenario["id"]: scenario}, [status], output)
    metrics = pd.read_csv(output / "notice_metrics.csv")
    np.testing.assert_allclose(metrics.first_price, 1.0)
    # Mutating a pre-disclosure action must fail the matched-history assertion.
    path = tmp_path / "test/trial/simultaneous_notice_public/rounds.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]["actions"][0] += 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="before disclosure"):
        analyze_notices(tmp_path, cfg, {scenario["id"]: scenario}, [status], output)


def test_cluster_pairing_preserves_distinct_model_pairs(cfg):
    rows = [
        {
            "dataset": "d",
            "scenario": f"s{s}",
            "direction": "forward",
            "seed_index": 0,
            "pair": pair,
            "arm": arm,
            "terminal_gain": value + (s / 10),
        }
        for s in range(3)
        for pair, values in [("a", (2, 1)), ("b", (5, 2))]
        for arm, value in zip(("left", "right"), values, strict=True)
    ]
    result = clustered_difference(pd.DataFrame(rows), "left", "right", cfg, 6)
    assert result["effect"] == pytest.approx(2.0)
    assert result["clusters"] == 3


def test_campaign_resume_retries_failed_block_and_preserves_deadline(tmp_path, monkeypatch):
    import theseus.campaign as campaign
    import theseus.campaign_analysis as analysis

    spec, base = load_campaign(project_root() / "configs/campaign.yaml")
    spec.update(pairs=[spec["pairs"][0]], target_seeds=2, hard_stop_utc=None)
    calls = []

    async def fake_pipeline(cfg, resume, **kwargs):
        calls.append((cfg["protocol"]["test_seed_start"], cfg["experiment"]))
        if len(calls) == 1:
            raise RuntimeError("Synthetic failure")
        return resume

    monkeypatch.setattr(campaign, "pipeline", fake_pipeline)
    monkeypatch.setattr(analysis, "analyze_campaign", lambda directory: None)
    destination = tmp_path / "campaign"
    asyncio.run(campaign.run_campaign(spec, base, destination=destination))
    first = load_json(destination / "campaign.json")
    assert complete_waves(first) == []
    assert calls == [(0, "planted"), (0, "neutral")]
    asyncio.run(campaign.run_campaign(spec, base, resume=destination))
    final = load_json(destination / "campaign.json")
    assert final["deadline_utc"] == first["deadline_utc"]
    assert complete_waves(final) == [0, 1]
    assert calls == [(0, "planted"), (0, "neutral"), (0, "planted"), (1, "planted"), (1, "neutral")]
    # An expired manifest cannot gain another allocation by resuming.
    final["deadline_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    final["blocks"] = final["blocks"][:2]
    atomic_json(destination / "campaign.json", final)
    asyncio.run(campaign.run_campaign(spec, base, resume=destination))
    assert len(calls) == 5


def test_aggregation_excludes_partial_waves_even_with_large_effects(tmp_path):
    from theseus.campaign_analysis import analyze_campaign

    spec, base = load_campaign(project_root() / "configs/campaign.yaml")
    spec.update(pairs=[spec["pairs"][0]], target_seeds=2)
    base["inference"]["backend"] = "mock"
    record = {"identity": {"spec": spec, "base": base, "source_hash": "fixture"}, "blocks": []}
    for wave, regime in [(0, "planted"), (0, "neutral"), (1, "planted")]:
        name = f"{wave}_{regime}"
        run = tmp_path / "studies" / name
        output = run / "analysis"
        output.mkdir(parents=True)
        rows = [
            {
                "dataset": "d",
                "scenario": f"s{s}",
                "direction": "forward",
                "seed_index": wave,
                "arm": arm,
                "terminal_gain": (1000 if wave else 1) if arm == "sequential_full" else 0,
            }
            for s in range(3)
            for arm in ("sequential_full", "sequential_joint_reset")
        ]
        pd.DataFrame(rows).to_csv(output / "episode_metrics.csv", index=False)
        record["blocks"].append(
            {
                "id": name,
                "wave": wave,
                "regime": regime,
                "pair": "a__b",
                "run_directory": str(run),
                "status": "complete",
            }
        )
    atomic_json(tmp_path / "campaign.json", record)
    analyze_campaign(tmp_path)
    effects = load_json(tmp_path / "analysis/primary_effects.json")
    assert all(r["effect"] == 1 for r in effects if r["available"])
    assert (tmp_path / "analysis/report_draft.pdf").read_bytes().startswith(b"%PDF")
    assert 150 <= len((tmp_path / "analysis/abstract_draft.txt").read_text().split()) <= 250
    all_rows = pd.read_csv(tmp_path / "analysis/all_branch_metrics.csv")
    assert (all_rows[all_rows.wave == 1].terminal_gain == 1000).any()
    assert not all_rows[all_rows.wave == 1].included_in_primary.any()
