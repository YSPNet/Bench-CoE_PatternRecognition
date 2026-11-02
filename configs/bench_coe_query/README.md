# Experiment contract guide

`experiment_contract.yaml` is the immutable source of truth for this isolated experiment. Runs copy it into their output directory and record its SHA-256.

## Provenance and status

`method` names the experiment and marks it `adapted`. The router behavior was recovered from exact hashes under the audited candidate `/home/sm5/ys/Bench-CoE`; because that directory has no frozen authoritative revision, the implementation must not claim `faithful`. `upstream_vlrb_commit` identifies the upstream code whose official data-building files match this checkout.

## Frozen data

`data` pins every split, Y, C, cost bounds, meta, and both registries by path, count/shape, and SHA-256. `benchmarks.manifest_sha256` is a deterministic manifest hash over the 14 BENCHMARKS JSONL files. `experts.canonical_order` is the only allowed matrix-column order. The adapter resolves rows through `sample_index.pkl`; directory iteration is never an ordering source.

The asset policy is a VL-RouterBench-matched adaptation. Packaged raw paths from the producer are not portable, so the adapter resolves each row from the official TSV using its canonical dataset/index. It returns one RGB image without resizing. The approved full-run overlay keeps the 47 genuine MMMU multi-image rows as structured errors: they are excluded from VLM feature/head loss, while frozen-denominator prediction uses the train-only StrongestGlobal expert and reports fallback coverage separately.

## Labels and evaluation

`labels` freezes the Bench-CoE query target: compute per-expert accuracy from train only, choose the highest-accuracy correct expert, and break a tie with the lower canonical index. All-wrong rows receive `-100` and do not contribute to cross entropy, but remain in routed evaluation.

`evaluation` separates prediction from scoring. Checkpoint selection uses dev hard-label loss only. Routed accuracy is the primary metric; cost and Rank Score use the official C/bounds and `routers.utils.rank_score` implementation. Hard-label accuracy is diagnostic and excludes all-wrong rows.

## Models and profiles

Each `models` entry pins a user-confirmed local directory plus config/processor/code hashes. Loading is local-only.

- BERT settings (native pooled/CLS classifier path, max length 256, full fine-tuning) come from the hash-audited Bench-CoE candidate.
- TinyLLaVA uses that candidate's explicit two-branch representation: mean of projected image tokens concatenated with the attention-masked mean of Phi-2's final text states. The exact SegR remote code creates `SiglipImageProcessor` locally at 384 pixels.
- Qwen3-VL uses the local native image+text chat template and the attention-masked mean of `model.model` final multimodal states.

`benchcoe_faithful_evidence` records source hashes but is not an executable faithful claim. `vlrb_adapted_v1` is the preregistered main profile. Its optimizer, epoch, patience, dropout, and dev-loss selection values reproduce the audited candidate behavior while adapting labels/data to VL-RouterBench.

## Reproducibility, caching, and scheduling

Seeds are exactly 0–4. `features` defines one-writer NumPy/JSONL shards and the fields that invalidate cache reuse. `scheduling` assigns deterministic row-modulo shards to eight independent GPU processes and maps five training seeds to GPUs 0–4. OOM ladders are recommendations only; a downgrade must be recorded and cannot silently change the contract.

## Safety gates

`RUN_API_REGEN` is permanently zero. `RUN_FULL` defaults to zero, which permits only the fixed 64/32/64 one-epoch smoke limits. A full run additionally requires the quality/extraction/parity gates, at least 20% free disk, all hashes unchanged, approval of adapted pooling/provenance, and a recorded multi-image policy. The contract remains byte-immutable; post-smoke decisions are captured in the contract-hash-bound `outputs/bench_coe_query/audit/FULL_RUN_APPROVAL.json`, whose hash is embedded in the quality gate and every full-run manifest.
