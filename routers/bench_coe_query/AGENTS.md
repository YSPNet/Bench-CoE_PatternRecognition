# Scoped experiment constraints

These rules apply to every file under `routers/bench_coe_query/`.

1. Treat `configs/bench_coe_query/experiment_contract.yaml` as immutable input. Verify its frozen data/model hashes before a run.
2. Never call an expert model, regenerate an expert answer/correctness value, or call any remote model/API. Never record credentials. Force Hugging Face/Transformers offline loading.
3. Never construct a new data split. Read exactly `SPLITS/train.jsonl`, `dev.jsonl`, and `test.jsonl`; fail if any file is absent, duplicated, overlapping, incomplete, or hash-mismatched.
4. Resolve matrix rows and model columns explicitly through the frozen registries. Do not infer ordering from directory traversal or use fabricated fallback IDs/costs.
5. Compute expert accuracies, hard labels, class/training statistics, and Strongest/Cheapest policies from train only. Dev selects checkpoints. Test is evaluation only.
6. Label a sample with the train-global-accuracy-best member of its correct expert set, breaking ties by the canonical column index. Give all-wrong rows `ignore_index=-100`; retain them in routed evaluation.
7. The main routed metric is `mean(Y[row, predicted_column])`. Read predictions from a frozen disk artifact before the evaluator accesses Y/C. Never alter predictions using Y/C.
8. Image resolution is strict: one decoded RGB image, no adapter-side resize, no white/black fallback. Return a structured error for missing, corrupt, empty, or multi-image samples; never select the first of multiple images.
9. BERT is fully trainable. TinyLLaVA and Qwen3-VL backbones are fully frozen and may only feed the hash-pinned adapted pooling in the contract. Do not call `.generate()`.
10. Checkpoint selection monitors only `dev_hard_label_loss`. Test metrics cannot select a checkpoint, seed, hyperparameter, processor, pooling rule, or batch size.
11. Feature shards use `row_index % world_size == rank`, one writer per file, atomic state updates, signature-bound resume, and an exact-once merge.
12. `RUN_API_REGEN` must equal `0`. When `RUN_FULL != 1`, reject any 30,540-row extraction or full five-seed training. Do not weaken gates after a failure.
