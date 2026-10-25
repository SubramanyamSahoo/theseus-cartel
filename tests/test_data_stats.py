import numpy as np
import pandas as pd
from theseus.analysis import clustered_difference, holm
from theseus.data import decode_sas_hex, scenarios_from_panel


def test_hex_exact():
    np.testing.assert_allclose(decode_sas_hex(pd.Series(["3FF0000000000000", "4000000000000000"])), [1, 2])


def test_cluster_not_rounds_and_paired_missingness(cfg):
    rows = [
        {
            "dataset": "a",
            "scenario": str(product),
            "direction": direction,
            "seed_index": seed,
            "arm": arm,
            "terminal_gain": float(arm == "left"),
        }
        for product in range(12)
        for direction in ["forward", "reverse"]
        for seed in range(3)
        for arm in ["left", "right"]
    ]
    frame = pd.DataFrame(rows)
    estimate = clustered_difference(frame, "left", "right", cfg, 3)
    assert estimate["clusters"] == 12 and estimate["matched_trials"] == 72
    assert estimate["effect"] == 1 and estimate["ci_low"] == 1
    less = frame[~((frame["scenario"] == "0") & (frame["arm"] == "right"))]
    assert clustered_difference(less, "left", "right", cfg)["clusters"] == 11
    assert holm([0.01, 0.03, 0.04]) == [0.03, 0.06, 0.06]


def test_product_splits_and_temporal_no_leakage(cfg):
    rows = [
        {
            "dataset": dataset,
            "product": str(product),
            "week": week,
            "quantity": 10.0,
            "price": 2.0 if week < 10 else 999.0,
            "cost": np.nan,
            "rows": 5,
            "description": "fixture",
        }
        for dataset in ["a", "b"]
        for product in range(10)
        for week in range(20)
    ]
    scenarios, splits = scenarios_from_panel(pd.DataFrame(rows), cfg)
    assert len(scenarios) == 16
    for s in scenarios:
        assert s["fit_median_price"] == 2 and s["heldout_median_price"] == 999
    for split in splits.values():
        assert not set(split["development"]) & set(split["test"])
