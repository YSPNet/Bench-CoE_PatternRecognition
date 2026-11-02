# Bench-CoE query routers

This isolated package implements the adapted VL-RouterBench query-routing experiment frozen in `configs/bench_coe_query/experiment_contract.yaml`.

The public CLI supports strict data audit, offline TinyLLaVA/Qwen3-VL feature extraction, BERT training, frozen-head training, Y/C-free prediction serialization, independent evaluation, the Prompt-10 quality gate, and Prompt-13 aggregation:

```bash
python -m routers.bench_coe_query.cli --help
bash scripts/bench_coe_query/preflight.sh
bash scripts/bench_coe_query/run_quality_gate.sh
bash scripts/bench_coe_query/run_full_experiment.sh
```

All model loaders are local-only. `RUN_API_REGEN` must remain `0`. Full operations are refused unless `RUN_FULL=1`, the contract-bound approval artifact is unchanged, `QUALITY_GATE.json` says `ready_for_full_run: true`, its recorded source files are unchanged, and current free disk remains above the frozen threshold. Prompt-12 training additionally requires passed, hash-bound extraction gates for both VLMs.

The smoke result is deliberately diagnostic: seed 0, one epoch, and fixed 64/32/64 train/dev/test rows. The full controller sequentially extracts TinyLLaVA then Qwen3-VL on eight deterministic shards, runs the three five-seed training groups, independently re-evaluates frozen predictions, and writes final tables under `outputs/bench_coe_query/final/`.
