import numpy as np
from scipy.optimize import minimize_scalar
from theseus.market import Market
from theseus.analysis import regret
from theseus.upstream import market_class


def test_original_parity_and_currency_invariance(cfg, scenario):
    original = market_class(cfg)(verbose=False)
    corrected = Market(cfg, scenario)
    np.testing.assert_allclose(corrected.nash, original.lower_prices, rtol=1e-8)
    np.testing.assert_allclose(corrected.monopoly, original.upper_prices, rtol=1e-8)
    action = np.array([1.5, 1.8])
    baseline = corrected.evaluate(action)
    scaled = Market(cfg, {**scenario, "price_scale": 13.7, "volume_scale": 231.0})
    result = scaled.evaluate(action * 13.7)
    np.testing.assert_allclose(result["gains"], baseline["gains"], rtol=1e-8)
    np.testing.assert_allclose(scaled.nash, corrected.nash * 13.7)
    np.testing.assert_allclose(result["demands"], np.array(baseline["demands"]) * 2.31)
    for i in range(2):
        prices = scaled.nash.copy()
        eps = np.cbrt(np.finfo(float).eps) * prices[i]
        prices[i] += eps
        right = scaled.counterfactual_record({}, prices)["profits"][i]
        prices[i] -= 2 * eps
        left = scaled.counterfactual_record({}, prices)["profits"][i]
        assert abs((right - left) / (2 * eps)) < 1e-5


def test_nash_monopoly_and_analytic_regret(cfg, scenario):
    market = Market(cfg, scenario)
    np.testing.assert_allclose(market.evaluate(market.nash)["gains"], [0, 0], atol=1e-10)
    np.testing.assert_allclose(market.evaluate(market.monopoly)["gains"], [1, 1], atol=1e-10)
    assert abs(regret(market, market.nash)) < 1e-10
    prices = market.monopoly
    numeric = []
    for i in range(2):

        def negative_profit(price, i=i):
            pair = prices.copy()
            pair[i] = price
            return -market.counterfactual_record({}, pair)["profits"][i]

        sol = minimize_scalar(negative_profit, bounds=(0, prices[i] * 2), method="bounded")
        actual = market.counterfactual_record({}, prices)["profits"][i]
        numeric.append((-sol.fun - actual) / (market.monopoly_profit[i] - market.nash_profit[i]))
    np.testing.assert_allclose(regret(market, prices), np.mean(numeric), atol=1e-8)


def test_counterfactual_recomputes_all_outcomes(cfg, scenario):
    market = Market(cfg, scenario)
    original = market.evaluate(market.monopoly)
    counter = market.counterfactual_record(original, market.nash)
    actual = market.evaluate(market.nash)
    for key in actual:
        np.testing.assert_allclose(counter[key], actual[key])
