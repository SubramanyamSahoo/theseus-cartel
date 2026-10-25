from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .common import (
    append_jsonl,
    atomic_json,
    digest,
    gather_clean,
    load_json,
    read_jsonl,
    stable_seed,
    utcnow,
)
from .inference import action
from .market import Market


def model_roles(direction):
    if direction == "forward":
        return "founder", "replacement"
    if direction == "reverse":
        return "replacement", "founder"
    raise ValueError("Unknown turnover direction")


class State:
    def __init__(self, cfg, scenario, parent, direction, arm, seed):
        self.cfg, self.arm, self.seed = cfg, arm, seed
        self.market = Market(cfg, scenario)
        self.old, self.new = model_roles(direction)
        self.roles = [self.old, self.old]
        self.views = [copy.deepcopy(parent), copy.deepcopy(parent)]
        self.first = seed % 2
        if arm == "fresh_final":
            self.roles = [self.new, self.new]
            self.views = [
                [self.market.counterfactual_record(r, self.market.nash) for r in parent] for _ in range(2)
            ]

    def before_round(self, local_round):
        p = self.cfg["protocol"]
        replace = []
        if (self.arm == "simultaneous_full" and local_round == 0) or (
            (self.arm == "simultaneous_delayed" or self.arm.startswith("simultaneous_notice_"))
            and local_round == p["turnover_gap"]
        ):
            replace = [0, 1]
        elif self.arm.startswith("sequential_"):
            if local_round == 0:
                replace = [self.first]
            elif local_round == p["turnover_gap"]:
                replace = [1 - self.first]
        events = []
        for agent in replace:
            before = [digest(v) for v in self.views]
            self.roles[agent] = self.new
            if self.arm == "sequential_drop_pre":
                self.views[agent] = []
            elif self.arm == "sequential_joint_reset":
                self.views = [[], []]
            elif self.arm == "sequential_competitive":
                self.views[agent] = [
                    self.market.counterfactual_record(r, self.market.nash) for r in self.views[agent]
                ]
            elif self.arm == "sequential_shuffle":
                window = self.cfg["market"]["history_window"]
                preserved = self.views[agent][:-window]
                original = self.views[agent][-window:]
                rng = np.random.default_rng(stable_seed(self.seed, "shuffle", agent, local_round))
                order = rng.permutation(len(original))
                # Joint records travel together; keep time labels ordered to remove time-format cues.
                self.views[agent] = preserved + [
                    {**original[j], "t": original[k]["t"]} for k, j in enumerate(order)
                ]
            events.append(
                {
                    "agent": agent,
                    "local_round": local_round,
                    "new_role": self.new,
                    "before_hashes": before,
                    "after_hashes": [digest(v) for v in self.views],
                }
            )
        return events

    def notice(self, local_round):
        if (
            not self.arm.startswith("simultaneous_notice_")
            or local_round < self.cfg["protocol"]["turnover_gap"]
        ):
            return None
        from .notices import notice_text

        return notice_text(self.arm.rsplit("_", 1)[1], self.seed)

    def scripted_price(self, agent, prefix):
        if prefix == "PC":
            return self.market.monopoly[agent]
        if self.cfg["inference"]["mock_policy"] == "history_copy" and self.views[agent]:
            # A deliberately nonstrategic rival-copying fixture, not an empirical model.
            return self.views[agent][-1]["actions"][1 - agent]
        return self.market.nash[agent]

    def intervene(self, local_round, proposed):
        prices = list(proposed)
        probe = self.cfg["protocol"]["deviation_probe"]
        event_round = self.cfg["protocol"]["turnover_gap"] + probe["after_full_turnover_rounds"]
        if self.arm != "sequential_deviation" or local_round != event_round:
            return prices, None
        agent = 1 - self.first
        previous = self.views[agent][-1]["actions"]
        prices[agent] = self.market.best_response(agent, previous[1 - agent])
        return prices, {
            "agent": agent,
            "responder": 1 - agent,
            "rule": probe["action_rule"],
            "local_round": local_round,
            "proposed_price": proposed[agent],
            "executed_price": prices[agent],
            "last_rival_price": previous[1 - agent],
            "undercuts_last_rival": prices[agent] < previous[1 - agent],
            "undercuts_current_rival": prices[agent] < prices[1 - agent],
        }

    def observe(self, row):
        for view in self.views:
            view.append(row)


async def episode(
    cfg, scenario, client, directory: Path, *, direction, seed, arm, rounds, parent=None, prefix=None
):
    parent = parent or []
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "scenario": scenario,
        "direction": direction,
        "seed": seed,
        "arm": arm,
        "rounds": rounds,
        "parent_hash": digest(parent),
        "prefix": prefix,
        "config_hash": digest(cfg),
    }
    meta = directory / "meta.json"
    if meta.exists():
        if load_json(meta)["identity"] != identity:
            raise ValueError(f"Resume identity mismatch: {directory}")
    else:
        atomic_json(meta, {"identity": identity, "started_utc": utcnow()})
    rows = read_jsonl(directory / "rounds.jsonl")
    if len(rows) > rounds:
        raise ValueError("Trajectory exceeds frozen length")
    state = State(cfg, scenario, parent, direction, arm, seed)
    offset = len(parent)
    for k, row in enumerate(rows):
        events = state.before_round(k)
        if row["events"] != events or row["roles"] != state.roles:
            raise ValueError("Stored trajectory does not match the intervention schedule")
        if row.get("notice") != state.notice(k):
            raise ValueError("Stored disclosure does not match the intervention schedule")
        prices, intervention = state.intervene(k, row["proposed_actions"])
        if intervention != row["intervention"] or prices != row["actions"]:
            raise ValueError("Stored probe does not match the frozen intervention")
        state.market.evaluate(row["actions"])
        state.observe(row)
    if len(rows) == rounds and (directory / "DONE.json").exists():
        if load_json(directory / "DONE.json")["trajectory_hash"] != digest(rows):
            raise ValueError("Stored trajectory checksum mismatch")
        return rows
    for k in range(len(rows), rounds):
        events = state.before_round(k)
        prefixes = [
            prefix
            or (cfg["protocol"]["founder_prefix"] if r == state.old else cfg["protocol"]["newcomer_prefix"])
            for r in state.roles
        ]
        prompts = [state.market.prompt(a, prefixes[a], state.views[a]) for a in range(2)]
        notice = state.notice(k)
        if notice:
            for prompt in prompts:
                prompt[-1]["content"] += "\n\nAdministrative notice:\n" + notice
        # Common random seeds across paired arms; no Python hash or scheduling-dependent seed.
        tasks = [
            client.choose(
                state.roles[a],
                prompts[a],
                stable_seed(seed, direction, scenario["id"], "round", offset + k, a),
                mock_price=state.scripted_price(a, prefixes[a]),
            )
            for a in range(2)
        ]
        replies = await gather_clean(*tasks)
        atomic_json(directory / "last_replies.json", {"local_round": k, "responses": replies})
        proposed = [action(r) for r in replies]
        prices, intervention = state.intervene(k, proposed)
        row = {
            "t": offset + k + 1,
            "local_round": k,
            **state.market.evaluate(prices),
            "roles": list(state.roles),
            "prefixes": prefixes,
            "proposed_actions": proposed,
            "intervention": intervention,
            "events": events,
            "responses": replies,
            "prompt_hashes": [digest(p) for p in prompts],
            "view_lengths": [len(v) for v in state.views],
            "notice": notice,
        }
        append_jsonl(directory / "rounds.jsonl", row)
        state.observe(row)
        rows.append(row)
    atomic_json(
        directory / "DONE.json",
        {"completed_utc": utcnow(), "rounds": len(rows), "trajectory_hash": digest(rows)},
    )
    return rows


def initial_history(cfg, scenario):
    market = Market(cfg, scenario)
    return [
        {"t": t + 1, **market.evaluate(market.nash)} for t in range(cfg["protocol"]["initial_nash_rounds"])
    ]


def established(rows, cfg):
    tail = rows[-cfg["protocol"]["establishment_window"] :]
    if not tail:
        return False
    return bool(
        np.min(np.mean([r["gains"] for r in tail], axis=0)) >= cfg["protocol"]["establishment_threshold"]
    )
