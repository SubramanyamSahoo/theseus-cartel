# The Cartel After Its Members - v3

**Full phased distribution:** start with `PHASES.md`. All original code, tests, documentation, and precomputed validation examples are included. After installing the basic Ubuntu tools and extracting this ZIP, run `bash START_FULL_TMUX.sh` to execute setup, validation, and the real campaign inside tmux. Monitoring and complete backup commands are included. The research implementation is unchanged.

**Can a pricing convention transfer across model families after every founder is replaced, and does a truthful replacement notice change it without deleting history?** This campaign extends the Theseus turnover experiment on the pinned Apart/Poseidon starter environment.

Four checkpoints from three families run as four model pairs, each in both replacement directions. Neutral founders and PC-instructed founders are separate conditions. Thirteen matched branches test history transfer, replacement timing, disclosures, and one forced best-response action. Three computational countermodels help interpret reactive behavior. The campaign uses a **24-hour wall-time budget**, with balanced seed waves and an absolute stop that reserves time for writing.

The package contains **software and labelled synthetic validation, not new GPU research findings**. Your completed v1 run stays a separate exploratory baseline. A successful execution gate does not show collusion, and longer computation cannot guarantee a positive result or a prize.

## Run on Lambda H100

Extract into a new directory; do not overwrite the code or results of a running v1/v2 experiment. The archive root is `theseus_cartel_h100_v3/`.

```bash
unzip theseus_v3_full_phased.zip
cd theseus_cartel_h100_v3
bash BOOTSTRAP.sh
.venv/bin/theseus campaign-plan
bash START_CAMPAIGN_TMUX.sh
tmux attach -t "$(cat logs/latest_session.txt)"
```

Detach with Ctrl-b, then d. The launcher prints the campaign directory and global deadline. It starts two localhost-only vLLM engines at a time on the single GPU, processes a model pair, and stops those engines before loading the next pair. Do not run the old suite at the same time on the same GPU/ports.

The stack pins vLLM 0.11.0, PyTorch 2.8.0 through its lock, and Transformers 4.57.1. Models use BF16. Four checkpoint downloads need roughly 85 GB of weight storage before dependency/data caches; check actual bytes in `validation/tokenizer_audit.json`. Reuse the existing Hugging Face cache automatically. Neither H100 startup nor real inference has been tested in the authoring environment.

`BOOTSTRAP.sh` needs Ubuntu with Python, git, tmux, and working NVIDIA drivers. If git/tmux are missing, it prints the installation command. It installs dependencies, verifies the pinned starter source, runs the tests, and prepares the two real retail datasets. For CPU-only validation, install `--extra dev --extra tokenizer` instead of `--extra gpu` using the locked environment.

### Models and comparisons

| Key | Pinned official checkpoint | Family |
|---|---|---|
| qwen8 | Qwen/Qwen3-8B | Qwen3 |
| qwen14 | Qwen/Qwen3-14B | Qwen3 |
| nemo12 | mistralai/Mistral-Nemo-Instruct-2407 | Mistral |
| olmo7 | allenai/OLMo-2-1124-7B-Instruct | OLMo2 |

Pairs: Qwen8/Qwen14, Qwen8/NeMo12, Qwen8/OLMo7, and NeMo12/OLMo7. Both directions run inside each block. These are feasible public checkpoints, not a claim to use the newest or strongest models. Native templates are retained; cross-family effects also change training and architecture. The common sampling settings are explicit design choices, not per-model optimal settings.

## Budget and balanced execution

`configs/campaign.yaml` sets a maximum of six seed waves and 24 hours. A wave includes every pair and both founder regimes: **8 blocks, 192 test founder trajectories, 192 separate test positive controls, and 2,496 post-founder branches** with the default 12 test contexts. It is not 2,496 independent economies. Development and test controls add requests. `campaign-plan` computes the workload from the configuration.

After the first complete wave, the next starts only if the longest completed wave plus 20% headroom fits the remaining time. The first wave has no measured runtime guarantee. A block failure stops additional waves after the current wave finishes its scheduled blocks. A time limit can leave a partial wave; its data are retained but excluded from pooled primary estimates. No time is spent idling merely to reach 24 hours.

The absolute stop is **26 October 2026 at 03:59 UTC / 09:29 IST**, eight hours before the verified submission deadline. The effective deadline is the earlier of this time and 24 hours from campaign creation. Shutdown/report generation may extend slightly beyond request cancellation. Resume preserves this deadline. The script stops its GPU processes; **it does not terminate the billed Lambda VM**.

Scientific settings are frozen per campaign. They include founder candidates [64,128,256], 64 post-turnover rounds, a 32-round terminal window, and 48-round positive controls. There is no globally optimal setting for an unknown behavioral effect. The provenance and limitations are documented in `docs/PARAMETERS.md`.

## Results and resume

```text
results/campaigns/<run-id>/
  campaign.json
  analysis/
    report_draft.pdf
    report.md
    abstract_draft.txt
    primary_effects.csv
    primary_effects.json
    primary_effects.png
    transfer_matrix.csv
    transfer_matrix.png
    coverage.csv
    all_branch_metrics.csv
    all_notice_metrics.csv
    all_test_controls.csv
  studies/<seed-pair-regime>/
    preregistration.json
    calibration.json
    COMPLETE.json
    logs/
    test/<trial>/<arm>/rounds.jsonl
    analysis/
      notice_first_action.png
      response_countermodels.png
      deviation_summary.json
      positive_controls.csv
      report_draft.pdf
```

`campaign.json` is execution status; the analysis folder contains scientific estimates. The component-study PDFs retain legacy within-block comparisons as diagnostics. The **campaign PDF and its six-comparison family** are the primary v3 analysis. Read control behavior and failures alongside estimates. The PDFs are review drafts; use the official submission template after reviewing results and writing the actual findings.

To resume exactly the same code/configuration:

```bash
bash START_CAMPAIGN_TMUX.sh --resume results/campaigns/<run-id>
```

To regenerate analysis from saved trajectories:

```bash
.venv/bin/theseus analyze-campaign results/campaigns/<run-id>
```

Complete blocks are skipped. Failed blocks reuse recorded trajectories and cached responses; invalid generations are never resampled until a desired answer appears. An expired deadline stays expired. Changes to code/configuration require a new campaign. Do not rename/move an active campaign before resuming it.

## Validate before long runs

```bash
.venv/bin/pytest -q
.venv/bin/theseus campaign-smoke
.venv/bin/python scripts/check_tokenizers.py
```

The campaign smoke runs all eight orchestration blocks with reduced synthetic contexts/horizons and a copying policy. It must recover the known planted-history effect and zero notice effect. It writes under `validation/`, never research `results/`. The tokenizer check downloads pinned tokenizer/config/index files, not weights. Neither validates GPU inference. Monitor the first real block's `logs/`, `calibration.json`, and `last_replies.json`; if PC controls cannot elicit high gains, interpret neutral nulls cautiously.

## Scope and research position

Recommended primary track: **3, Collusion Carriers and Countermeasures**. Secondary: **1, Markets and Collusion**. The proposed contribution is a controlled replacement-and-disclosure experiment, not merely more models, longer runs, or another memory ablation. See `docs/CAMPAIGN_PROTOCOL.md` for six hypotheses, prior-work comparison, interpretation of nulls, and the institutional-persistence framing.

The supplied `llm-cartel/environment/oligopoly.py` is imported from commit `e6fb1308dae9cf97ae42aec1c5f7889d9e2291e8`; it computes demand and profits. Our normalized-price solver corrects its missing 1/alpha derivative. Source hashes and compatibility checks prevent silent upstream drift. The scheduler implements the extra history and turnover interventions. Upstream source and raw retail records are fetched, not redistributed in this ZIP.

UCI Online Retail II and Dominick's oatmeal data supply product descriptions, scales, and available cost proxies. Demand remains assumed. Twelve products do not establish twelve independent real economies; this limitation is part of every report. No scratchpad, hidden-state transfer, model training, empirical demand estimation, or real-world cartel labels are included.

Legacy single-pair commands (`run`, `suite`, `START_SUITE_TMUX.sh`) remain available for separately labelled studies; `configs/h100.yaml` uses the shorter v2-style protocol under the v3 software schema. Never resume a v1/v2 run using v3 code. Keep the original v1 results archive for analysis alongside, not pooled into, this campaign.
