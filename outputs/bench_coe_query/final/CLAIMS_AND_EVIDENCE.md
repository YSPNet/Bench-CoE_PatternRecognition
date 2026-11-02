# Claims and evidence

- Run completeness: `RUN_COMPLETENESS.json` (15/15).
- Main routed/cost results: `MAIN_TABLE.csv` and `COST_TABLE.csv`.
- Per-dataset and difficulty evidence: `PER_DATASET.csv` and `DIFFICULTY_STRATA.csv`.
- Paired inference: `STATISTICAL_TESTS.csv`; unavailable comparisons are retained as negative evidence.
- Every independent metric artifact is under `independent_metrics/` and binds the frozen prediction SHA-256.
- `PREDICTION_PROVENANCE.json` binds all 45 prediction files to logits argmax, seed, checkpoint bytes, metrics/resource anchors, and frozen SHA-256 values.
- `INDEPENDENT_METRICS_MANIFEST.json` hash-gates all 15 complete independently regenerated test metric documents under the same pinned NumPy runtime.
- `CACHE_INTEGRITY.json` binds both merged caches through their external pointer/gate/manifest and all 16 shard payload digests.
- `INPUT_CONTENT_PROVENANCE.json` binds the approval overlay, all local model/tokenizer payloads, all 18 TSV image containers, 15 run manifests, and both feature-cache signatures.
- `REPRODUCIBILITY_BOUNDARY.json` distinguishes exact archived-artifact derivation from fresh-run/cross-geometry bitwise reproducibility.
- The provenance-remediation block in `REPRODUCIBILITY_BOUNDARY.json` binds the 16-entry legacy AMP migration ledger and all 16 full-state VLM rank attestations.
- Adapted (not faithful) provenance and pooling decisions remain disclosed in the audit directory.
