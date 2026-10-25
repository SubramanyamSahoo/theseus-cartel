import json

import numpy as np

from theseus.hardware import GIB, capacity_plan


def test_capacity_from_architecture_and_weight_sizes(cfg, tmp_path, monkeypatch):
    import theseus.hardware as h

    monkeypatch.setattr(
        h,
        "gpu_inventory",
        lambda: {
            "name": "NVIDIA H100 80GB",
            "total_bytes": 80 * GIB,
            "used_bytes": 0,
            "driver_version": "synthetic",
        },
    )

    def download(model, name, revision):
        large = "14B" in model
        value = (
            {"metadata": {"total_size": (28 if large else 16) * GIB}}
            if "index" in name
            else {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 40 if large else 32,
            }
        )
        target = tmp_path / (("large_" if large else "small_") + name)
        target.write_text(json.dumps(value))
        return str(target)

    monkeypatch.setattr(h, "hf_hub_download", download)
    plan = capacity_plan(cfg)
    items = plan["models"].values()
    assert all(x["max_num_seqs"] >= 1 for x in items)
    np.testing.assert_allclose(sum(x["gpu_memory_utilization"] for x in items), 0.9)
    assert all(x["kv_cache_bytes"] == 11 * GIB for x in items)
