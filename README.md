# Bench-CoE: final Pattern Recognition manuscript archive

This directory is a selective backup of the experimental code, frozen router
artifacts and measured results for `paper/Manuscript.pdf` (the final paper
provided by the authors on 2026-11-02). It is arranged as a future GitHub
repository. `MANIFEST.csv` gives the SHA-256 hash and size of every archived
source file. Run `python verify_release.py` to check the copy and rescore the
paper's five-seed two-state spherical router.

## Where the paper's results are

| Paper result | Code | Results and trained artifacts |
| --- | --- | --- |
| Query-Bench-CoE, cost-free and cost-aware TinyLLaVA | `scripts/journal_20261024/v19_query_softlabels_frozen.py`, `e20_query_softlabels.py`, `routers/bench_coe_query/` | `outputs/journal_20261024/V19_QUERY_COST_TINYLLAVA/` (five final heads, selected actions and source CV) |
| Subject-Bench-CoE, TinyLLaVA | `scripts/run_bench_coe_subject_mmmu.py`, `scripts/journal_20261024/e21_subject_cost_protocol.py` | `outputs/bench_coe_subject_mmmu30/ood11_3/tinyllava/` and `outputs/journal_20261024/E21/` |
| State-Bench-CoE, spherical K=2, five seeds | `scripts/journal_20261024/v23_mm_spherical_compare.py` | `outputs/journal_20261024/V23_MM_SPHERICAL_COMPARE/` |
| State count K=1,2,4,6,8,10,12; shrinkage and cost rules | `scripts/journal_20261024/v25_spherical_experiments.py` | `outputs/journal_20261024/V25_SPHERICAL_COMPLETION/` (final cluster models, frozen actions, CV and tables) |
| Encoder ablations | `scripts/journal_20261024/v23_state_modality_compare.py` | `outputs/journal_20261024/V23_STATE_MODALITY_COMPARE/` |
| Matched granularity, map stability and post-hoc control | `e12_matched.py`, `e15_stability.py`, `e16_guard.py` and their dependencies | `outputs/journal_20261024/E12/`, `E15/`, `E16/`, with supporting `E3/`, `E4/`, `E5/` |
| Paired significance analysis | `scripts/journal_20261024/v25_spherical_bootstrap.py` | `outputs/journal_20261024/V25_SPHERICAL_COMPLETION/PAIRED_BOOTSTRAP.csv` |
| Reference-router local replays | `scripts/journal_20261024/e6_*.py`, `routers/table_a2/` and baseline architectures | `outputs/journal_20261024/E6/` (source-selected seed-0 actions and summaries) |

The paper's Table 1 mixes published VL-RouterBench aggregate values with local
replays. Published aggregates are citation values, not model checkpoints in
this archive. The source-selected, frozen per-query actions and local summary
files preserve the distinction. The subject prototype uses MMMU training inputs;
the paper's MMMU OOD caveat still applies.

## Included inputs and limits

`SPLITS/`, `data/registry/` and the compact `Y`/`C` evaluation matrices are
included so frozen actions can be independently rescored. Full benchmark
prompts/images, Qwen3-VL/TinyLLaVA/BGE/SigLIP pretrained weights and large
feature caches are omitted. Obtain VL-RouterBench and the pretrained models
from their original distributors, then regenerate the feature caches before
refitting the routers. The frozen contract and archived scripts retain original
hashes and some machine-specific paths for provenance; configure these paths
for a new environment. This is an audited research snapshot, not a turnkey
training environment on a different machine.

The archive contains trained *router* heads, centroids and maps. It contains no
expert-model weights, API credentials, temporary image caches, old manuscript
drafts or obsolete experiment renderings. No expert inference is required to
rescore the included frozen actions.

## Quick check

```bash
python -m pip install -r requirements.txt
python verify_release.py
```

The verifier uses only NumPy and the included scoring function. For refitting,
use the entry points above after restoring the benchmark/feature inputs. The
source-only cost-knob selection and five seeds are documented in each experiment
`AUDIT.json` and `SOURCE_CV*.json`.
