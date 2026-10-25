# Full phased package: start here

This ZIP contains the complete v3 implementation, all original test/configuration/documentation files, every packaged synthetic trajectory and report, and the phase tools below. Nothing from the original full bundle was removed. Research source and configuration are unchanged. Downloadable model weights and source datasets are fetched during setup/execution; they are not embedded in the ZIP.

## Phase 0: upload and start from an empty workspace

Download `theseus_v3_full_phased.zip` to your computer and upload it to `/home/ubuntu/` on your Lambda H100 80 GB instance. For Windows PowerShell, replace the key path and IP in:

```powershell
scp -i "C:\path\your-lambda-key.pem" "$env:USERPROFILE\Downloads\theseus_v3_full_phased.zip" ubuntu@YOUR_LAMBDA_IP:/home/ubuntu/
```

In the Lambda terminal:

```bash
sudo apt-get update
sudo apt-get install -y git tmux unzip python3-venv util-linux
cd /home/ubuntu
unzip theseus_v3_full_phased.zip
cd theseus_cartel_h100_v3
sha256sum --check --quiet MANIFEST.sha256
bash START_FULL_TMUX.sh
tmux attach -t "$(cat logs/latest_full_session.txt)"
```

The launcher starts tmux **before** installation. It runs phases 1, 2, and 3, then prints phase 4 status. Setup errors stop the sequence. The ZIP extracts to `theseus_cartel_h100_v3/`; do not overwrite an active run's directory. Approximately 85 GB are required for checkpoint weights alone, plus dependencies, datasets, results, and backup space.

Detach with Ctrl-b, release, then d. Reattach from the project directory using the command above. Logs are under `logs/`. The launcher prints the exact log path even if a session exits before you attach.

## Phase 1: environment, starter code, tests, data

`phases/01_SETUP.sh` checks the GPU and storage, then invokes the original `BOOTSTRAP.sh`. Bootstrap installs the locked environment, fetches the pinned Apart/Poseidon starter, runs pytest, and prepares UCI/Dominick's data. Existing caches are reused when available. There is no need for an old project or old results.

Manual command inside tmux: `bash phases/01_SETUP.sh`.

## Phase 2: workload and synthetic validation

`phases/02_VALIDATE.sh` prints the exact campaign plan, checks that its absolute deadline has not passed, verifies all pinned tokenizer templates, and runs the full reduced synthetic campaign. It inspects the resulting manifest and every block's completion record. A zero process exit code alone is not treated as validation success. Synthetic results are not empirical model findings.

Manual command after phase 1: `bash phases/02_VALIDATE.sh`.

## Phase 3: real experiment

`phases/03_RUN.sh` invokes the original real campaign. Model weights download as needed. Four checkpoints, four pairs, both replacement directions, both founder regimes, and all thirteen branches remain configured. Two vLLM engines share the single GPU; model pairs run sequentially.

Manual command after validation: `bash phases/03_RUN.sh`.

The 24-hour campaign budget starts when the real campaign is created, **after setup and validation**. The existing absolute cutoff remains 2026-10-26 at 03:59 UTC / 09:29 IST. The earlier limit wins. The launcher does not extend deadlines or guarantee a full 24-hour runtime. First GPU startup and model inference remain unvalidated in the authoring environment.

## Phase 4: monitor and inspect results

From another Lambda terminal:

```bash
cd /home/ubuntu/theseus_cartel_h100_v3
python3 phases/04_STATUS.py
tail -n 50 "$(cat logs/latest_campaign_log.txt)"
```

The status tool follows the exact campaign directory from this launch's log. It does not guess from the newest filesystem folder. During setup there may be no real campaign yet. To inspect a particular run:

```bash
python3 phases/04_STATUS.py --run results/campaigns/YOUR_RUN_ID
```

Review `campaign.json` for stop reason and block statuses. The CLI can exit normally after a budget cutoff or failed blocks; that does not mean all planned work completed. Research estimates and review drafts are under `results/campaigns/YOUR_RUN_ID/analysis/`. Partial-wave data remain visible but are excluded from pooled primary estimates.

For an interrupted run with the same source/configuration and remaining budget, bypass setup and smoke:

```bash
bash START_FULL_TMUX.sh --resume results/campaigns/YOUR_RUN_ID
```

Resume preserves the original deadline. If the environment alone was deleted, reinstall it with phase 1 before resuming. If the recorded campaign itself was deleted, it cannot be resumed without its backup.

## Phase 5: export and download before deleting anything

After the campaign has stopped:

```bash
python3 phases/05_EXPORT.py results/campaigns/YOUR_RUN_ID
```

This creates a uniquely named `.tar.gz` under `/home/ubuntu/theseus_backups/` and a SHA256 checksum. It includes all campaign files, SQLite response caches, and the project source/configuration needed to retain provenance. It checks the actual campaign lock and refuses to snapshot an actively written campaign. A `.partial` file is not a completed backup. No data or files are deleted by export.

Download the printed archive and checksum to your computer before deleting the project or terminating the VM. Model caches and dependency environments are not included. Restoring a resumable run requires the original absolute project/run paths and the same source/configuration; the time budget does not reset. The script stops inference processes when the campaign ends, but Lambda billing continues until you terminate the VM.

## Included validation and limits

The complete prior validation example is retained under `validation/example_campaign/`. The original research implementation passed 39 tests and a synthetic eight-block campaign. This packaging adds orchestration and backup tools without changing the experiment. See `validation/PHASE_TOOL_CHECKS.md` for the additional checks. Generated reports are review drafts; no new real GPU findings are bundled or implied.
