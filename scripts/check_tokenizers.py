"""CPU-only check of pinned templates and shard metadata; downloads no model weights."""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from theseus.campaign import load_campaign, make_config
from theseus.common import project_root
from theseus.data import prepare
from theseus.hardware import measure_context_tokens


def main():
    spec, base = load_campaign(project_root() / "configs/campaign.yaml")
    scenario_path = prepare(base)
    scenarios = json.loads(Path(scenario_path).read_text())
    audit = {}
    for pair in (["qwen8", "qwen14"], ["nemo12", "olmo7"]):
        cfg = make_config(base, spec, pair, "planted", 0)
        measured = measure_context_tokens(cfg, scenarios)
        for role, key in zip(("founder", "replacement"), pair, strict=True):
            model = spec["models"][key]
            config = json.loads(
                Path(hf_hub_download(model["id"], "config.json", revision=model["revision"])).read_text()
            )
            index = json.loads(
                Path(
                    hf_hub_download(model["id"], "model.safetensors.index.json", revision=model["revision"])
                ).read_text()
            )
            audit[key] = {
                **model,
                "architecture": config["architectures"],
                "weight_bytes": index["metadata"]["total_size"],
                "weight_shards": len(set(index["weight_map"].values())),
                "tokens": measured[role],
            }
            print(key, audit[key], flush=True)
    output = project_root() / "validation/tokenizer_audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "scope": "Pinned official tokenizer and weight metadata checks on CPU; no weights loaded and no GPU inference validated.",
                "models": audit,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
