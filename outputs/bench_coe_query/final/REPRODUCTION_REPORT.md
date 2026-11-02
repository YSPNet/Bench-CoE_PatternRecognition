# Bench-CoE Query Router reproduction report

Contract SHA-256: `a1c24102e63954371a113d247d4beb72fdb03fc16bcd871d0589aa2399fa5e0f`.

All 15 preregistered runs are complete and were independently re-evaluated from frozen test prediction JSONL files. Checkpoint selection used dev hard-label loss only.

## Main results

| Method | Routed accuracy mean ± std | Delta to Strongest | Oracle-gap closure |
|---|---:|---:|---:|
| Query-BERT | 0.788343 ± 0.003567 | 0.005108 | 0.029518 |
| Query-TinyLLaVA | 0.788212 ± 0.004732 | 0.004977 | 0.028761 |
| Query-Qwen3VL | 0.789129 ± 0.002906 | 0.005894 | 0.034059 |
| StrongestGlobal | 0.783235 ± 0.000000 | 0.000000 | 0.000000 |
| Oracle | 0.956287 ± 0.000000 | 0.173052 | 1.000000 |

## Fairness and limitations

- Query-BERT is text-only; TinyLLaVA and Qwen3-VL are multimodal and still require their frozen backbones online. Head-only throughput is not end-to-end throughput.
- The 47 genuine multi-image rows are excluded from VLM feature/head training. Their frozen-denominator predictions use StrongestGlobal selected from train-only statistics; coverage is recorded in every metrics file.
- The evaluator requires the exact expected fallback set even when empty; all 45 prediction files passed this fail-closed policy check.
- All 45 router prediction files were frozen by target-free producer processes whose schemas contain no Y/C fields. Hash-bound receipts and filtered OS open traces show that the producer processes never opened Y.npz or C.npy; evaluators ran only after those processes exited.
- The AF-004 legacy diagnostic migration is preserved as a per-field hash-bound ledger. Unrecoverable historical AMP skipped-step status is explicitly unknown, while future training records skipped-step counts, scaler transitions, and typed gradient status.
- The AF-006 freeze probes hash every ordered state_dict parameter and persistent buffer from raw bytes before and after actual forward execution; all 16 independent rank-process attestations passed and are bound through the quality gate.
- `INPUT_CONTENT_PROVENANCE.json` content-addresses every selected local model/tokenizer payload and all 18 TSV image containers. Archived feature bytes were rebound without claiming a fresh extraction rerun.
- The 17 expert outcomes, including GPT-4o and Gemini, come only from official offline artifacts. No API or expert inference was run.
- Official MLP/RouterDC/VLC sample-level predictions are not present in this release, so paired tests against them are explicitly marked unavailable rather than fabricated.
- This is ID sample-level generalization over the same 14-dataset mixed pool, not held-out-dataset OOD evaluation.
- Archived predictions/metrics are byte-anchored and exactly derivable. Fresh BERT training and TinyLLaVA cross-geometry extraction are claimed numerically reproducible only, not bitwise identical; `REPRODUCIBILITY_BOUNDARY.json` records the boundary and the fail-closed policy for future runs.
