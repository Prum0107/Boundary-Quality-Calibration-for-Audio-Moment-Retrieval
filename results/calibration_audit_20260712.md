# Five-Seed Ranking and Calibration Audit

Date: 2026-07-12
Host: QD-DETR with the corrected detached-target BQC-Dec implementation
Evaluation: CASTELLA development-testing, 1,347 queries and 10 candidates per query
Seeds: 2026, 2027, 2028, 2029, 2030

## Protocol

All ranking and probability comparisons use the same BQC-Dec checkpoints and
candidate spans. The binary calibration target is `IoU >= 0.7`. Expected
calibration error uses 10 equal-width bins. The empirical-prevalence reference
predicts each seed's observed candidate positive rate and is a descriptive
class-imbalance reference, not a deployable model.

## Ranking on Identical Candidate Spans

| Measure | BQC quality | Host confidence |
|---|---:|---:|
| R1@0.7 | 17.76 +/- 0.27 | 16.30 +/- 0.24 |
| R1@0.5 | 28.05 +/- 0.43 | 26.62 +/- 0.39 |

The five-seed mean R1@0.7 difference is +1.45 percentage points. The 95%
Student-t interval over paired seed differences is [+1.11, +1.80].

## Candidate-Level Calibration

| Probability / estimate | Mean +/- std |
|---|---:|
| BQC IoU regression MAE | 0.1630 +/- 0.0078 |
| BQC qcls Brier | 0.0385 +/- 0.0012 |
| Host confidence Brier | 0.3122 +/- 0.0030 |
| Empirical-prevalence Brier reference | 0.0449 +/- 0.0012 |
| BQC qcls ECE-10 | 0.0201 +/- 0.0017 |
| Host confidence ECE-10 | 0.3532 +/- 0.0034 |

The BQC qcls score improves Brier error over both host confidence and the
prevalence reference. The prevalence reference has approximately zero ECE by
construction, which is why ECE is interpreted with Brier score and ranking
performance rather than alone.
