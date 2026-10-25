import asyncio
import copy
import json

import numpy as np
import pytest

from theseus.cli import smoke_config
from theseus.common import load_config, load_json, project_root
from theseus.inference import Client
from theseus.market import Market
from theseus.protocol import State, episode


def test_separate_planted_profile():
    neutral = load_config(project_root() / "configs/h100.yaml")
    planted = load_config(project_root() / "configs/planted.yaml")
    assert neutral["experiment"] == "neutral" and neutral["protocol"]["founder_prefix"] == "P0"
    assert planted["experiment"] == "planted" and planted["protocol"]["founder_prefix"] == "PC"
    assert planted["protocol"]["newcomer_prefix"] == "P0"
    assert planted["models"] == neutral["models"]
    assert planted["execution"]["results_root"] != neutral["execution"]["results_root"]


def test_delayed_simultaneous_matches_full_turnover_time(cfg, scenario):
    sequential = State(cfg, scenario, [], "forward", "sequential_full", 0)
    delayed = State(cfg, scenario, [], "forward", "simultaneous_delayed", 0)
    gap = cfg["protocol"]["turnover_gap"]
    for t in range(gap + 1):
        sequential.before_round(t)
        delayed.before_round(t)
        if t < gap:
            assert sequential.roles == ["replacement", "founder"]
            assert delayed.roles == ["founder", "founder"]
    assert sequential.roles == delayed.roles == ["replacement", "replacement"]


def test_probe_is_single_action_no_lookahead_and_resumes(cfg, scenario, tmp_path):
    cfg, _ = smoke_config(cfg)
    cfg["inference"]["mock_policy"] = "history_copy"
    market = Market(cfg, scenario)
    parent = [{"t": i + 1, **market.evaluate(market.monopoly)} for i in range(3)]

    async def check():
        client = Client(cfg, tmp_path / "cache.sqlite")
        try:
            shared = dict(direction="forward", seed=2, rounds=6, parent=parent)
            reference = await episode(
                cfg, scenario, client, tmp_path / "reference", arm="sequential_full", **shared
            )
            probe = await episode(
                cfg, scenario, client, tmp_path / "probe", arm="sequential_deviation", **shared
            )
            shocks = [r for r in probe if r["intervention"]]
            assert len(shocks) == 1
            shock = shocks[0]["intervention"]
            t = shock["local_round"]
            a = shock["agent"]
            assert [r["actions"] for r in reference[:t]] == [r["actions"] for r in probe[:t]]
            assert probe[t]["actions"][1 - a] == reference[t]["actions"][1 - a]
            np.testing.assert_allclose(
                shock["executed_price"], market.best_response(a, probe[t - 1]["actions"][1 - a])
            )
            assert shock["undercuts_current_rival"]
            partial = tmp_path / "partial"
            partial.mkdir()
            (partial / "meta.json").write_bytes((tmp_path / "probe/meta.json").read_bytes())
            (partial / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in probe[: t + 1]))
            resumed = await episode(cfg, scenario, client, partial, arm="sequential_deviation", **shared)
            assert resumed == probe
            # Only the forced round is externally controlled; subsequent actions are model proposals.
            assert all(r["actions"] == r["proposed_actions"] for r in probe[t + 1 :])
        finally:
            await client.close()

    asyncio.run(check())


def test_scripted_pipeline_recovers_known_nonzero_effect(cfg, tmp_path):
    from theseus.runner import pipeline

    cfg, scenarios = smoke_config(cfg)
    cfg["experiment"] = "planted"
    cfg["protocol"]["founder_prefix"] = "PC"
    cfg["inference"]["mock_policy"] = "history_copy"
    cfg["execution"]["results_root"] = str(tmp_path / "scripted")
    run = asyncio.run(pipeline(cfg, prepared_scenarios=scenarios))
    effects = load_json(run / "analysis/effects.json")
    result = next(r for r in effects if r["right"] == "sequential_joint_reset")
    # PC returns monopoly, neutral copies the last rival observation, and an empty view returns Nash.
    # Retained histories stay at monopoly; a synchronized reset leaves both sellers at Nash.
    assert result["effect"] == pytest.approx(1.0, abs=1e-8)
    assert result["ci_low"] == pytest.approx(1.0, abs=1e-8)
    assert result["ci_high"] == pytest.approx(1.0, abs=1e-8)
    assert len(load_json(run / "analysis/positive_controls.json")) == 8
    assert load_json(run / "analysis/deviation_summary.json")["paired_trials"] == 8
    assert (run / "analysis/report_draft.pdf").read_bytes().startswith(b"%PDF")
    assert 150 <= len((run / "analysis/abstract_draft.txt").read_text().split()) <= 250


def test_failed_test_controls_fail_quality_gate_without_dropping_branches(cfg, tmp_path, monkeypatch):
    import theseus.runner as runner

    cfg, scenarios = smoke_config(cfg)
    cfg["execution"]["results_root"] = str(tmp_path / "failed_controls")
    original = runner.episode

    async def fail_control(c, s, client, directory, **kwargs):
        if directory.name == "positive_control":
            raise ValueError("Synthetic invalid positive-control completion")
        return await original(c, s, client, directory, **kwargs)

    monkeypatch.setattr(runner, "episode", fail_control)
    with pytest.raises(RuntimeError, match="Failure-rate quality gate"):
        asyncio.run(runner.pipeline(cfg, prepared_scenarios=scenarios))
    run = next((tmp_path / "failed_controls").iterdir())
    complete = load_json(run / "COMPLETE.json")
    assert complete["failed_arms"] == 0
    assert complete["planned_arms"] == 80
    assert complete["failed_test_positive_controls"] == 8
    assert not complete["failure_rate_gate_passed"]
    assert len(load_json(run / "analysis/positive_controls.json")) == 8


def test_measured_context_increases_capacity_without_changing_context_limit(cfg, tmp_path, monkeypatch):
    import theseus.hardware as h

    monkeypatch.setattr(
        h,
        "gpu_inventory",
        lambda: {
            "name": "NVIDIA H100 80GB",
            "total_bytes": 80 * h.GIB,
            "used_bytes": 0,
            "driver_version": "fixture",
        },
    )

    def download(model, name, revision):
        p = tmp_path / (model.split("/")[-1] + name)
        spec = (
            {"metadata": {"total_size": 22 * h.GIB}}
            if "index" in name
            else {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
            }
        )
        p.write_text(json.dumps(spec))
        return str(p)

    monkeypatch.setattr(h, "hf_hub_download", download)
    worst = h.capacity_plan(cfg)
    measured = h.capacity_plan(cfg, {role: {"reference_context_tokens": 1800} for role in cfg["models"]})
    for role in cfg["models"]:
        assert measured["models"][role]["max_num_seqs"] > worst["models"][role]["max_num_seqs"]
        assert measured["models"][role]["maximum_context_tokens"] == 4096


def test_suite_continues_separate_study_and_resumes_only_failed(cfg, tmp_path, monkeypatch):
    import theseus.suite as suite

    neutral = copy.deepcopy(cfg)
    planted = copy.deepcopy(cfg)
    planted["experiment"] = "planted"
    planted["protocol"]["founder_prefix"] = "PC"
    calls = []

    async def fake_pipeline(c, resume):
        calls.append(c["experiment"])
        if len(calls) == 1:
            raise RuntimeError("Synthetic first-study failure")
        return resume

    def create(c):
        p = tmp_path / c["experiment"]
        p.mkdir(exist_ok=True)
        return p

    monkeypatch.setattr(suite, "pipeline", fake_pipeline)
    monkeypatch.setattr(suite, "new_run", create)
    target = tmp_path / "suite"
    with pytest.raises(RuntimeError, match="failed studies"):
        asyncio.run(suite.run_suite([neutral, planted], target))
    assert calls == ["neutral", "planted"]
    asyncio.run(suite.run_suite([neutral, planted], target))
    assert calls == ["neutral", "planted", "neutral"]
    assert all(r["status"] == "complete" for r in load_json(target / "suite.json")["studies"])
