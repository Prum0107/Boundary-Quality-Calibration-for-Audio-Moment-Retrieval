<div align="center">

# Boundary Quality Calibration for Audio Moment Retrieval

**Train candidate scores to reflect temporal boundary quality, not only foreground confidence.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-research_code-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-059669.svg)](LICENSE)
[![Target audit](https://img.shields.io/badge/IoU_targets-detached-2563EB.svg)](docs/CODE_AUDIT.md)
[![Seeds](https://img.shields.io/badge/evaluation-5_seeds-7C3AED.svg)](results/mechanism_ablation_20260711.md)

</div>

## Why BQC?

Audio Moment Retrieval (AMR) receives a long audio recording and a natural-
language query, then predicts the start and end time of the matching moment.
DETR-style AMR systems generate several candidate windows and commonly rank
them by foreground confidence. Foreground confidence is not trained to estimate
temporal Intersection over Union (IoU), so the most confident candidate may not
have the best boundaries.

On the CASTELLA development-testing split, the frozen QD-DETR host reaches an
R1@0.7 of **14.33%**, while Oracle@10 reaches **28.88%**. Better candidates
often exist in the candidate pool but are not ranked first.

**Boundary Quality Calibration (BQC-Dec)** adds an explicit quality head to
complete decoder slots and trains the ranking signal with three detached
targets:

1. continuous IoU regression;
2. IoU@0.7 correctness classification;
3. within-query list-wise ordering.

## Method

<p align="center">
  <img src="assets/bqc_method_overview.png" alt="BQC-Dec method overview" width="920">
</p>

During BQC-Dec fine-tuning:

| Component | Status |
|---|---|
| Audio-text encoder | Frozen |
| Learned temporal reference points (`query_embed`) | Frozen |
| Existing transformer decoder | **Trainable** |
| Existing span/class readout heads | Frozen |
| New quality head | **Trainable** |

The decoder architecture and span readout interface are unchanged. Frozen
readout parameters still pass the original host-loss gradients into the
decoder. IoU targets are computed from current predicted spans and annotations
under `stop-gradient`; no ground-truth window is used at inference.

## Main Results

All main numbers use one unified evaluator on CASTELLA development-testing
(1,347 queries). Checkpoints, hyperparameters, and score rules are selected on
development-validation. Results are means over seeds 2026--2030.

### Mechanism study

| Variant | Trainable scope | R1@0.7 | Gain vs frozen host | Spearman |
|---|---|---:|---:|---:|
| Frozen QD-DETR host | None | 14.33 | - | 0.239 |
| Quality head only | Quality head | 16.27 +/- 0.06 | +1.95 | 0.367 |
| Reference points + quality head | Reference points, quality head | 16.38 +/- 0.12 | +2.05 | 0.369 |
| Decoder + host loss | Decoder | 16.20 +/- 0.31 | +1.87 | 0.237 |
| Decoder + quality only | Decoder, quality head | 10.32 +/- 1.88 | -4.01 | 0.366 |
| Decoder + host + shuffled quality | Decoder, quality head | 13.63 +/- 1.70 | -0.70 | 0.313 |
| **BQC-Dec: host + detached quality** | **Decoder, quality head** | **17.76 +/- 0.27** | **+3.43** | **0.392** |

BQC-Dec improves over matched decoder adaptation by **+1.56 R1@0.7
percentage points**. All five paired seed differences are positive; the 95%
Student-t interval is **[+1.17, +1.95]**.

### Same-candidate ranking and calibration

The following ranking comparison uses identical BQC-Dec checkpoints and
predicted spans; only the ranking score changes.

| Measure | BQC quality | Host confidence / reference |
|---|---:|---:|
| R1@0.7 | **17.76 +/- 0.27** | 16.30 +/- 0.24 |
| R1@0.5 | **28.05 +/- 0.43** | 26.62 +/- 0.39 |
| IoU regression MAE | **0.163 +/- 0.008** | - |
| IoU@0.7 Brier score | **0.0385 +/- 0.0012** | confidence: 0.3122 +/- 0.0030 |
| Empirical-prevalence Brier reference | **0.0449 +/- 0.0012** | - |
| IoU@0.7 ECE (10 bins) | **0.0201 +/- 0.0017** | confidence: 0.3532 +/- 0.0034 |

Quality ranking adds **+1.45 percentage points** over confidence ranking on
identical spans. The Brier result also beats the empirical-prevalence reference,
so the low error is not explained only by the approximately 4.7% positive rate.

## Evaluation Scope

DCASE 2026 names the CASTELLA train, validation, and test splits
development-training, development-validation, and development-testing. This
repository reports CASTELLA development-testing results. The separate
100-recording DCASE challenge evaluation set has no public temporal annotations
and is not evaluated here.

Current evidence is limited to **QD-DETR on CASTELLA**. A CG-DETR audio
adaptation did not produce a usable candidate pool, so cross-host
generalization remains untested. BQC-Dec improves selection among generated
candidates; it cannot recover a queried event missing from the candidate pool
and does not directly move boundaries at inference.

## Repository Layout

```text
.
|-- assets/                       README method figure and editable source
|-- src/
|   |-- baseline/                 QD-DETR host source
|   |-- bqc/                      current detached-target implementation
|   `-- legacy_non_detached/      superseded source retained for provenance
|-- patches/
|   |-- qd_detr_bqc_detached.patch
|   `-- qd_detr_bqc_legacy.patch
|-- scripts/
|   |-- run_mechanism_ablation.py
|   `-- legacy_reproduction/      historical wrappers with server paths
|-- tools/audit_parameters.py
|-- tests/test_source_invariants.py
|-- audits/                       machine-readable parameter audit
|-- results/                      verified result summaries
`-- docs/                         architecture, experiment map, and code audit
```

The legacy non-detached implementation is retained only to document the
project history. Current claims and tables use the corrected detached-target
rerun.

## Using the Code

This repository contains the BQC source, patches, controls, and audits. The
official AMR dataset features and model checkpoints are not redistributed.

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Apply BQC to a compatible QD-DETR AMR checkout

Copy the current source:

```bash
cp src/bqc/qd_detr.py /path/to/amr_host/src/qd_detr.py
```

or inspect and apply the patch:

```bash
git apply patches/qd_detr_bqc_detached.patch
```

### 3. Audit checkpoint parameter changes

```bash
python tools/audit_parameters.py \
  --base /path/to/host_checkpoint.pth \
  --bqc /path/to/bqc_checkpoint.pth \
  --output parameter_audit.json
```

The legacy reproduction wrappers retain their original absolute server paths
for provenance and require local path adaptation before use.

## Reproducibility Notes

- [Detached-target mechanism results](results/mechanism_ablation_20260711.md)
- [Five-seed ranking and calibration audit](results/calibration_audit_20260712.md)
- [Experiment map](docs/EXPERIMENT_MAP.md)
- [Architecture notes](docs/ARCHITECTURE.md)
- [Code audit findings](docs/CODE_AUDIT.md)
- [Parameter audit](audits/parameter_audit_seed2026.json)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## References

- N. Carion et al., “End-to-End Object Detection with Transformers,” ECCV,
  2020. [Paper](https://arxiv.org/abs/2005.12872)
- W. Moon et al., “Query-Dependent Video Representation for Moment Retrieval
  and Highlight Detection,” CVPR, 2023.
  [Paper](https://arxiv.org/abs/2303.13874) ·
  [Code](https://github.com/wjun0830/QD-DETR)
- H. Munakata et al., “Language-based Audio Moment Retrieval,” ICASSP, 2025.
  [Paper](https://arxiv.org/abs/2409.15672)
- H. Munakata et al., “CASTELLA: Long Audio Dataset with Captions and Temporal
  Boundaries,” 2026. [Paper](https://arxiv.org/abs/2511.15131)

## Acknowledgements

This work builds on DETR, Moment-DETR, QD-DETR, and the DCASE 2026 Audio Moment
Retrieval baseline. Their open-source implementations and task design made this
study possible.
