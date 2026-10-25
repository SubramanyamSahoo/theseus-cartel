# Draft bug report: currency scaling in the equilibrium solver

Repository: https://github.com/poseidon-research/collusion_hackathon

Commit: `e6fb1308dae9cf97ae42aec1c5f7889d9e2291e8`

File: `llm-cartel/environment/oligopoly.py`; methods `foc` and `foc_monopoly`.

Demand uses `(a_i - p_i/alpha)/mu`, but the equilibrium derivative terms divide markups by `mu` without the additional `alpha`. At non-unit currency scales, computed Nash and monopoly benchmarks therefore fail scale invariance even though the payoff calculation scales prices and costs.

The symmetric-game Nash FOC is `1 - (p_i-alpha*c_i)*(1-s_i)/(alpha*mu) = 0`. The monopoly markup derivatives also require `alpha*mu`. Equivalently, solve normalized prices `x_i=p_i/alpha` and convert back:

- Nash: `1 - (x_i-c_i)*(1-s_i)/mu = 0`.
- Monopoly: `1 - (x_i-c_i)/mu + sum_j((x_j-c_j)*s_j)/mu = 0`.

Regression tests compare the original symmetric unit-scale solution, test currency/volume invariance, and check numerical profit derivatives:

```bash
.venv/bin/pytest -q tests/test_economics.py
```

The extension keeps the imported payoff step and overrides benchmark solving. Observation-prompt changes are disclosed separately. This draft has not been posted or sent to organizers.
