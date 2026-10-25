# Parameter provenance and tuning

## v3 campaign overrides

The table later in this file describes the shorter single-pair profiles in `h100.yaml` / `planted.yaml`. `campaign.yaml` and `campaign_base.yaml` override them as follows. Every non-derived value is a disclosed design or engineering choice; none is claimed globally optimal.

| Campaign setting | Value / rule | Justification and limit |
|---|---|---|
| Models | Qwen3-8B, Qwen3-14B, Mistral-NeMo-12B, OLMo-2-7B Instruct; pinned revisions | Three families, public access, architectural support in pinned vLLM; not the newest models. |
| Pair set | Qwen8/Qwen14, Qwen8/NeMo12, Qwen8/OLMo7, NeMo12/OLMo7, both directions | Fixed four-pair design: within-family reference plus cross-family transfers. Not all six possible unordered pairs. |
| Sampling | Common .7/.8/20 temperature/top-p/top-k; native chat templates | Fixed across conditions, not optimal for every family. Qwen alone receives enable_thinking=false. |
| Founder candidates / fallback | [64,128,256] / 128 | Longer opportunity for formation; selected using development only. |
| Development / test PC controls | 48 rounds each | Covers the larger terminal window; includes test controls regardless of founder formation. |
| Follow-up / terminal score | 64 / 32 rounds | Declared persistence window. No inference about indefinite stability. |
| Notices | Silent, administrative placebo, private/public replacement | Same simultaneous turnover round and inherited observations; wording/salience confounds remain. |
| Notice variants | Three prewritten variants; trial seed modulo three | Reproducible assignment, not guaranteed balanced under a limited seed budget. |
| Computational countermodels | Copy rival, hold own, exact myopic best response | Parameter-free policy definitions, zero added model calls; not fitted mechanistic explanations. |
| Test seeds | One per full wave, sequential offsets, up to six waves | Keeps model/regime coverage balanced. Repeated seeds remain nested observations. |
| Primary multiplicity | Six comparisons across both founder regimes | Three endpoints per regime; Bonferroni interval tails and Holm p-values across six. |
| Campaign wall budget | 24 hours from creation | Loading/calibration included; persists across resume, unlike legacy per-invocation budgets. |
| Hard stop | 2026-10-26 03:59 UTC | Reserves eight hours for report/submission; effective stop is the earlier limit. Shutdown may take additional time. |
| New-wave reserve | Longest complete wave duration x1.20 | Explicit throughput uncertainty allowance, not a completion guarantee. No outcome-based stopping. |

Tokenizer measurement includes every notice variant on full development histories. Weight downloads use exactly the shard files in the pinned safetensors index, avoiding duplicate consolidated checkpoint files. Existing response caching, finite concurrency benchmarking, structural economics, source checksums, and failure rules remain in force.

## Shared settings and shorter single-pair defaults

No study can certify every parameter as globally optimal. This project makes the distinction explicit: economic assumptions, research design choices, numerical tolerances, and measured engineering settings have different justifications. The complete YAML is copied into every run. Changing it requires a new experiment.

| Configuration | Default / rule | Reason and limitation |
|---|---|---|
| Model pair | Official Qwen3-8B and Qwen3-14B, pinned commits | Different checkpoints within one family; two BF16 engines are intended to fit one H100. Not a claim that these are the strongest available models. |
| Precision | BF16 weights and KV | Avoid an undeclared quantization confound. Capacity is estimated and startup-tested. |
| Qwen sampling | temperature .7, top-p .8, top-k 20, non-thinking | Official non-thinking guidance. Fixed across arms; not tuned to maximize collusion. |
| Output tokens | 96 | Capacity allowance for one short numeric answer. Truncation fails; it is not repaired. |
| Structured price output | On | vLLM regex constrains answer syntax to the price line, including scientific notation. It sets no finite upper price bound and does not select an action. Record this decoding constraint when comparing with unconstrained studies. |
| Context | 4096 tokens; 32 visible rounds | Weekend compute/context compromise, not an optimal memory length. Requests exceeding the server context fail visibly. |
| Economic `a0`, qualities, costs, `mu` | 0, [2,2], [1,1], .25 | Structural parameters from the supplied symmetric game. Dominick's costs override the cost vector with acquisition-cost proxies. |
| Price/quantity scale | Earlier-half product medians | Computed from data. UCI scale maps the structural Nash price to the observed median. Dominick's scale is observed median price. |
| Equilibrium tolerance | 1e-8 normalized FOC residual | Numerical accuracy criterion; validated by scale invariance and finite differences. |
| Prompt significant digits | 8 | Explicit formatting precision to preserve scale. Not an economic rounding rule. |
| Product eligibility | >=20 weeks and >=30 valid source rows | Data-coverage design choice, fixed before LLM outcomes. |
| Temporal split | First .5 for calibration | Predeclared descriptive calibration; held-out price summaries are not demand validation. |
| Product split | 2 development + 6 test per dataset | Deterministic hash allocation, disjoint products. Twelve test clusters are modest; no guaranteed power. |
| Seed | 20261025 | Reproducibility namespace based on the sprint date. All per-request seeds are hash-derived. |
| Development repetitions | 2 seeds × both directions | Practical calibration budget. |
| Test repetitions | 3 seeds × both directions | Paired Monte Carlo repetitions nested within context. |
| Founder instruction | P0 in neutral; PC in planted; successors P0 | Two predeclared, separately labelled studies. Never select a study based on test success. |
| Founder horizon | Select from [32,64,128] on development | Shortest maximizing established study-founder histories; 64 fallback if no formation. Test results do not tune it. |
| Establishment | Both mean gains >=.5 over 20 trailing rounds | Operational half-monopoly-gain benchmark; affects development selection and exploratory subset only. |
| Positive control | `PC`, 24 rounds in development and each test trial | Explicitly instructed control, retained regardless of founder behavior and separate from main estimates. |
| Initial history | One Nash observation | Shared, disclosed neutral initialization. No hidden collusive demonstration. |
| Replacement gap | 16 rounds | Declared adaptation interval; neither learned nor claimed optimal. |
| Primary simultaneous timing | At second sequential replacement, round 16 | Matches full-turnover time; immediate simultaneous replacement remains secondary. |
| Follow-up | 32 rounds after full turnover; last 16 scored | Declared horizon and averaging window. Temporal persistence beyond this is untested. |
| Deviation action | Exact best response to last rival price | Computed from structural economics, not a handpicked price cut. No lookahead; undercut is not guaranteed. |
| Probe timing / windows | 8 rounds after full turnover; 8 baseline, 4 response, 8 recovery | Declared diagnostic windows, not tuned to test trajectories; recovery uses the final post-probe rounds. |
| Total VRAM allocation | .90 of measured total | Explicit reserve for driver/runtime uncertainty. Not a proven memory bound. |
| Engine overhead allowance | 3 GiB per engine | Conservative planning input for activations/graphs; actual logs determine fit. |
| KV bytes | Remaining memory / number of engines | Derived from measured device capacity, checkpoint bytes, and overhead allowance. |
| Reference context | Largest tokenized full development history plus 96 output tokens, times 1.25, capped at 4096 | Pinned tokenizer measurement, both agents, Nash/monopoly histories and all instruction prefixes. 25% headroom is explicit, not a proven upper bound on all future prompts. |
| Server max sequences | min(128, KV bytes / per-reference-context KV bytes) | Architecture-derived estimate; longer live requests may cause preemption. Token scheduling remains vLLM's job. Doctor uses conservative 4096 when no measured histories are available. |
| Batched prefill tokens | 8192 | Explicit engineering starting point. Chunked prefill handles queued work. Not claimed to be the fastest value. |
| CPU threads per engine | 4 | Avoid CPU oversubscription. Adjust before a new run for your machine. |
| Concurrent episodes | 128 | Bounded orchestration cap. Individual model request limits independently constrain GPU demand. |
| Concurrent requests per model | Best measured among [8,16,32,64] | Both engines active; 2 repetitions/candidate after warmup; fastest fully valid candidate wins. Search is finite and noisy. |
| Startup/request timeouts | 900 / 180 seconds | Operational failure boundaries. Transport retries=2, exponential backoff base=1 second. No numerical-output-dependent retry. |
| Response caching | On | Reuse exact request/revision/backend/seed responses on resume. |
| Alpha | .05 | Declared inferential error convention. Three primary contrasts per study use adjusted intervals and Holm p-values. Six cross-study claims are not jointly corrected by those per-study families. |
| Bootstrap MC SE | .005 | Determines 10,000 draws using a probability-error bound. Does not prove interval coverage. |
| Minimum test clusters | 12 | Reporting gate, not a power analysis. Fewer contexts remain exploratory. |
| Failure gate | .05 of planned branches and .05 of test PC controls, separately | Execution-quality policy; all missing records remain disclosed. Success is not a guarantee of control behavior. |
| Wall-time budget | 12 hours per study invocation | Suite may use two allocations. User-editable soft local cutoff. Resume resets wall time; an absolute cost cutoff does not reset if billing start is unchanged. |
| Hourly price / cost ceiling | null until supplied | Never invent a Lambda price. Cost enforcement requires real price and timezone-aware billing start. |

Small numerical constants in algorithms (two sellers, two KV tensors, BF16 bytes, the Lambert-W formula, significance-family size) arise from definitions or architecture. Plot sizes and display precision affect presentation only. The code does not hardcode successful outcomes, selected favorable test products, collusion labels, or rescue prices.
