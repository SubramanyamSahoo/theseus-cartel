import asyncio
import copy
import json

import httpx
import pytest

from theseus.common import gather_clean
from theseus.inference import Budget, BudgetExceeded, Client, action
from theseus.protocol import State


def test_http_model_identity_and_invalid_cache(cfg, tmp_path):
    async def check():
        client = Client(cfg, tmp_path / "requests.sqlite")
        await client.http.aclose()
        calls = []

        def handler(request):
            calls.append(str(request.url))
            assert "regex" in json.loads(request.content)["structured_outputs"]
            return httpx.Response(
                200,
                json={
                    "model": cfg["models"]["founder"]["id"],
                    "choices": [{"message": {"content": "invalid price"}, "finish_reason": "stop"}],
                },
            )

        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
        messages = [{"role": "user", "content": "synthetic test"}]
        try:
            reply = await client.choose("founder", messages, 12)
            with pytest.raises(ValueError):
                action(reply)
            again = await client.choose("founder", messages, 12)
            assert again == reply and len(calls) == 1
            with pytest.raises(RuntimeError, match="Returned model"):
                await client.choose("replacement", messages, 12)
            assert ":8100/" in calls[0] and ":8101/" in calls[1]
        finally:
            await client.close()

    asyncio.run(check())


def test_sibling_requests_cancel_before_client_shutdown():
    async def check():
        cancelled = asyncio.Event()

        async def long_request():
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()

        async def failure():
            await asyncio.sleep(0)
            raise BudgetExceeded("synthetic budget interruption")

        with pytest.raises(BudgetExceeded):
            await gather_clean(long_request(), failure())
        assert cancelled.is_set()

    asyncio.run(check())


def test_budget_enforces_wall_time(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["execution"]["max_wall_hours"] = 0
    with pytest.raises(BudgetExceeded):
        Budget(cfg).check()


def test_duplicate_inflight_requests_are_coalesced(cfg, tmp_path):
    cfg = copy.deepcopy(cfg)
    cfg["inference"]["backend"] = "mock"

    async def check():
        client = Client(cfg, tmp_path / "coalesced.sqlite")
        try:
            replies = await asyncio.gather(
                *[
                    client.choose("founder", [{"role": "user", "content": "fixture"}], 3, mock_price=1.5)
                    for _ in range(8)
                ]
            )
            assert all(r == replies[0] for r in replies)
            assert client.requests == 1 and client.hits == 7
        finally:
            await client.close()

    asyncio.run(check())


def test_shuffle_preserves_visible_set_and_joint_records(cfg, scenario):
    cfg = copy.deepcopy(cfg)
    cfg["market"]["history_window"] = 4
    parent = [
        {
            "t": i + 1,
            "actions": [i, i + 1],
            "profits": [i * 3, i * 5],
            "demands": [i + 2, i + 7],
            "gains": [i / 2, i / 3],
        }
        for i in range(12)
    ]
    state = State(cfg, scenario, parent, "forward", "sequential_shuffle", 8)
    state.before_round(0)
    view = state.views[0]
    assert view[:-4] == parent[:-4]
    assert sorted(tuple(r["actions"]) for r in view[-4:]) == sorted(tuple(r["actions"]) for r in parent[-4:])
    for row in view[-4:]:
        original = next(r for r in parent if r["actions"] == row["actions"])
        assert row["profits"] == original["profits"] and row["demands"] == original["demands"]
    assert [r["t"] for r in view] == list(range(1, 13))
