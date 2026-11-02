# E6 acceptance or execution failure

Recorded: 2026-10-28 08:48:31 CST

Final E6 status: `acceptance_needs_recheck`. The strict original gate queue was replaced with a v7 continuation that requires the independent split/scorer/feature recheck and keeps the discrepancy labeled.

## Five-seed 14ID best-test checks

| Method | Observed Rank | Paper Rank | Absolute gap | Within 2 points |
| --- | ---: | ---: | ---: | --- |
| mlp | 75.125 | 74.230 | 0.895 | True |
| routerdc | 71.477 | 74.590 | 3.113 | False |
| vlc | 74.783 | 74.330 | 0.453 | True |

## Independent split and feature recheck

The 14ID seed-0 RouterDC Rank gap is not caused by UID split, Y/C alignment, cost bounds, saved-action scoring or missing strict image/SigLIP feature coverage. Training behavior and release-versus-paper configuration remain possible causes.

Released-code commit: `f5a41ce83d6efe67489b35bb7a07ae72976a9aba`.

The split, matrix order, Rank bounds and saved-action replay checks are recorded in `E6/RECHECK_PRELIM.json`. This recheck was made after the first complete RouterDC seed; it does not erase a five-seed failure.

E6/SUMMARY.csv and E6/AUDIT.json retain all measured values. No missing result was estimated or copied.
