import asyncio
import copy
import json
import numpy as np
import pytest
from theseus.common import read_jsonl
from theseus.inference import Client, action, parse_price
from theseus.protocol import State, episode, initial_history


def test_history_reset_is_once_and_turnover_is_real(cfg, scenario):
    parent = initial_history(cfg, scenario)
    state = State(cfg, scenario, parent, "forward", "sequential_drop_pre", 4)
    state.before_round(0)
    assert state.roles == ["replacement", "founder"]
    assert state.views[0] == [] and state.views[1]
    row = {"t": 2, **state.market.evaluate(state.market.nash)}
    state.observe(row)
    state.before_round(1)
    assert state.views[0] == [row]
    state.before_round(cfg["protocol"]["turnover_gap"])
    assert state.roles == ["replacement", "replacement"]
    assert state.views[0] and not state.views[1]


def test_joint_reset_and_competitive_history(cfg, scenario):
    parent = initial_history(cfg, scenario) * 4
    for arm in ["sequential_joint_reset", "sequential_competitive", "simultaneous_full", "fresh_final"]:
        state = State(cfg, scenario, parent, "reverse", arm, 5)
        state.before_round(0)
        state.before_round(cfg["protocol"]["turnover_gap"])
        assert state.roles == ["founder", "founder"]
        if arm == "sequential_joint_reset":
            assert state.views == [[], []]
        if arm == "sequential_competitive":
            assert [len(v) for v in state.views] == [4, 4]
            for view in state.views:
                for row in view:
                    np.testing.assert_allclose(row["gains"], [0, 0], atol=1e-10)


@pytest.mark.parametrize(
    "text",
    [
        "NaN",
        "My chosen price: nan",
        "My chosen price: -1",
        "My chosen price: 1e999",
        "Price: 2",
        "My chosen price: 1\nActually 2",
    ],
)
def test_invalid_actions_fail(text):
    with pytest.raises(ValueError):
        parse_price(text)


def test_truncation_fails():
    with pytest.raises(ValueError):
        action({"finish_reason": "length", "text": "My chosen price: 2"})


def test_resume_matches_uninterrupted_and_cache(cfg, scenario, tmp_path):
    cfg = copy.deepcopy(cfg)
    cfg["inference"]["backend"] = "mock"
    cfg["protocol"]["turnover_gap"] = 2

    async def run():
        client = Client(cfg, tmp_path / "cache.sqlite")
        try:
            kwargs = dict(
                direction="forward",
                seed=2,
                arm="sequential_drop_pre",
                rounds=5,
                parent=initial_history(cfg, scenario),
            )
            full = await episode(cfg, scenario, client, tmp_path / "full", **kwargs)
            partial = tmp_path / "partial"
            partial.mkdir()
            (partial / "meta.json").write_bytes((tmp_path / "full/meta.json").read_bytes())
            (partial / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in full[:2]))
            resumed = await episode(cfg, scenario, client, partial, **kwargs)
            assert full == resumed
            assert client.hits == 6
        finally:
            await client.close()

    asyncio.run(run())


def test_torn_jsonl_only_final_record_is_recovered(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"n":1}\n{"n":')
    assert read_jsonl(path) == [{"n": 1}]
    path.write_text('{bad}\n{"n":1}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)
