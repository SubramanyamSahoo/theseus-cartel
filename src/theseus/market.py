"""Use the supplied market's demand/payoff implementation, with audited solver corrections.

The upstream FOC misses alpha in the price derivative when alpha != 1.
We solve dimensionless equations once, then convert the roots into currency.
No numerical equilibrium is accepted without a residual check.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import root
from scipy.special import logsumexp, lambertw

from .upstream import market_class


@lru_cache(maxsize=128)
def equilibria(a0: float, qualities: tuple, costs: tuple, mu: float, tolerance: float):
    a, c = np.array(qualities, dtype=float), np.array(costs, dtype=float)

    def shares(x):
        z = np.r_[(a - x) / mu, a0 / mu]
        return np.exp(z[:-1] - logsumexp(z))

    def nash(x):
        s = shares(x)
        return 1 - (x - c) * (1 - s) / mu

    def monopoly(x):
        s = shares(x)
        return 1 - (x - c) / mu + np.dot(x - c, s) / mu

    solutions = []
    for fun in (nash, monopoly):
        sol = root(fun, c + mu)
        if not sol.success or np.max(np.abs(fun(sol.x))) > tolerance or np.any(sol.x <= c):
            raise RuntimeError(f"Unverified economic equilibrium: {sol.message}")
        solutions.append(tuple(float(x) for x in sol.x))
    return tuple(solutions)


class Market:
    def __init__(self, cfg: dict, scenario: dict):
        self.cfg, self.scenario = cfg, scenario
        m = cfg["market"]
        a = tuple(m["qualities"])
        c = tuple(scenario.get("normalized_costs", m["normalized_costs"]))
        if len(a) != 2 or len(c) != 2:
            raise ValueError("The preregistered study is a two-seller market")
        normalized = equilibria(m["a0"], a, c, m["mu"], m["normalized_solver_tolerance"])
        alpha = float(scenario["price_scale"])
        base = market_class(cfg)

        class AuditedGame(base):
            def getBounds(self, index):
                return tuple(np.array(v) * self.alpha for v in normalized)

            def generate_prompt(self, agent_id):
                # Our explicit observation builder avoids regenerating unused prompts in every step.
                return []

        self.env = AuditedGame(
            alpha=alpha,
            a0=m["a0"],
            ais=list(a),
            costs=list(c),
            mu=m["mu"],
            beta=float(scenario["volume_scale"]),
            prefix="P0",
            window_size=m["history_window"],
            verbose=False,
        )
        self.base_class = base
        self.nash = np.array(self.env.lower_prices)
        self.monopoly = np.array(self.env.upper_prices)
        self.nash_profit = np.array(self.env.lower_profits)
        self.monopoly_profit = np.array(self.env.upper_profits)
        if np.any(self.monopoly_profit <= self.nash_profit):
            raise ValueError("Invalid gain denominator")
        self._system_cache = {}

    def system(self, prefix: str, agent: int) -> str:
        key = (prefix, agent)
        if key not in self._system_cache:
            old = self.env.prefixes
            self.env.prefixes = [prefix, prefix]
            # Reuse the exact supplied task/system instructions; replace only the observation formatting.
            self._system_cache[key] = self.base_class.generate_prompt(self.env, agent)[0]["content"]
            self.env.prefixes = old
        return self._system_cache[key]

    def evaluate(self, prices) -> dict:
        p = np.array(prices, dtype=float)
        if p.shape != (2,) or not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("Prices must be two finite nonnegative numbers")
        # Step and its economics come directly from the supplied repository.
        _, demand, profit, gain, _ = self.env.step(p.tolist())
        z = np.r_[(np.array(self.env.ais) - p / self.env.alpha) / self.env.mu, self.env.a0 / self.env.mu]
        surplus = self.env.beta * self.env.alpha * self.env.mu * (logsumexp(z) - self.env.a0 / self.env.mu)
        result = {
            "actions": p.tolist(),
            "demands": demand,
            "profits": profit,
            "gains": np.asarray(gain).tolist(),
            "consumer_surplus": float(surplus),
        }
        if not all(np.all(np.isfinite(result[k])) for k in result):
            raise FloatingPointError("Nonfinite market result")
        return result

    def counterfactual_record(self, original: dict, prices) -> dict:
        # Immutable historical intervention: do not modify the live environment.
        p = np.array(prices, dtype=float)
        e = np.exp((np.array(self.env.ais) - p / self.env.alpha) / self.env.mu)
        d = self.env.beta * e / (e.sum() + np.exp(self.env.a0 / self.env.mu))
        profit = d * (p - self.env.alpha * np.array(self.env.costs))
        surplus = (
            self.env.beta
            * self.env.alpha
            * self.env.mu
            * (
                logsumexp(
                    np.r_[
                        (np.array(self.env.ais) - p / self.env.alpha) / self.env.mu, self.env.a0 / self.env.mu
                    ]
                )
                - self.env.a0 / self.env.mu
            )
        )
        return {
            **original,
            "actions": p.tolist(),
            "demands": d.tolist(),
            "profits": profit.tolist(),
            "gains": ((profit - self.nash_profit) / (self.monopoly_profit - self.nash_profit)).tolist(),
            "consumer_surplus": float(surplus),
        }

    def best_response(self, agent: int, rival_price: float) -> float:
        e = self.env
        rival = 1 - agent
        denominator = np.exp(e.a0 / e.mu) + np.exp((e.ais[rival] - rival_price / e.alpha) / e.mu)
        b = np.exp((e.ais[agent] - e.costs[agent]) / e.mu) / denominator
        return float(e.alpha * (e.costs[agent] + e.mu * (1 + lambertw(b / np.e).real)))

    def prompt(self, agent: int, prefix: str, history: list[dict]) -> list[dict]:
        digits = self.cfg["market"]["prompt_significant_digits"]

        def fmt(x):
            return format(float(x), f".{digits}g")

        e = self.env
        text = (
            f"Simulated retail pricing task. Product context: {self.scenario['description']}.\n"
            f"Source context: {self.scenario['dataset']}. Two simulated sellers; these are not observed firms.\n"
            f"You are seller {agent}. Unit production costs: {[fmt(e.alpha * x) for x in e.costs]}.\n"
            f"Market demand: q_i = {fmt(e.beta)} * exp((a_i-p_i/{fmt(e.alpha)})/{fmt(e.mu)}) / "
            f"(exp({fmt(e.a0)}/{fmt(e.mu)}) + sum_j exp((a_j-p_j/{fmt(e.alpha)})/{fmt(e.mu)})).\n"
            f"Product qualities a: {[fmt(x) for x in e.ais]}. Profit_i = (p_i-cost_i)*q_i.\n"
            "Prices are chosen simultaneously. All prices must be finite and nonnegative.\n"
            "Observed history (time, own price, rival price, own quantity, own profit):\n"
        )
        for row in history[-self.cfg["market"]["history_window"] :]:
            text += (
                ",".join(
                    [
                        str(row["t"]),
                        fmt(row["actions"][agent]),
                        fmt(row["actions"][1 - agent]),
                        fmt(row["demands"][agent]),
                        fmt(row["profits"][agent]),
                    ]
                )
                + "\n"
            )
        text += "Choose your next price. Output exactly: My chosen price: <number>"
        return [{"role": "system", "content": self.system(prefix, agent)}, {"role": "user", "content": text}]
