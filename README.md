<div align="center">

# Boundary Quality Calibration for Audio Moment Retrieval

**IoU-aware candidate scoring for DETR-style temporal audio grounding**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-research_code-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-059669.svg)](LICENSE)
[![Audit](https://img.shields.io/badge/parameter_scope-audited-2563EB.svg)](audits/parameter_audit_seed2026.json)

</div>

## Overview

Audio Moment Retrieval (AMR) receives an untrimmed audio recording and a
natural-language query, then returns the start and end time of the matching
moment. DETR-style hosts generate a fixed set of candidate windows and usually
rank them by foreground confidence. That score is not necessarily calibrated
to temporal Intersection over Union (IoU).

Boundary Quality Calibration (BQC) studies a narrower question:

> Given a generated candidate pool, can the ranking score be trained to track
> temporal boundary quality more faithfully?

The main variant, **BQC-Dec**, attaches a quality head to the final transformer
decoder states and supervises candidate quality from three views:

1. continuous IoU regression;
2. IoU@0.7 correctness classification;
3. within-query list-wise ordering.

## Method at a Glance

```mermaid
flowchart LR
    A[Long audio + text query] --> E[Query-dependent encoder]
    Q[Learned moment queries / anchors] --> D[Existing transformer decoder]
    E --> D
    D --> H[Decoder slot states]
    H --> S[Frozen span and class heads]
    H --> B[Added quality head]
    S --> W[Candidate start, end, confidence]
    B --> R[IoU regression]
    B --> C[IoU@0.7 classification]
    B --> L[Within-query ordering]
    B --> K[Quality-ranked top-1]
```

Under the audited BQC-Dec freeze policy:

| Component | Status during BQC fine-tuning |
|---|---|
| Moment queries / trainable anchors | Frozen |
| Transformer encoder | Frozen |
| Transformer decoder layers | Trainable |
| Span and class heads | Frozen |
| Quality head | Trainable, newly added |

The method fine-tunes the **existing decoder layers**. It does not redesign the
decoder architecture and does not adapt the moment anchors in the reported
configuration. See [Architecture Notes](docs/ARCHITECTURE.md) and the
[Parameter Audit](docs/CODE_AUDIT.md).

## Audited Historical Results

Results below use QD-DETR on CASTELLA dev-test and five seeds. The two BQC-Dec
rows evaluate identical checkpoints and predicted spans; only the ranking score
changes.

| Method | R1@0.7 | Gain vs host | Spearman | Pairwise accuracy |
|---|---:|---:|---:|---:|
| QD-DETR confidence | 14.03 | - | 0.2320 | 0.6493 |
| IoU-aware confidence baseline | 16.60 | +2.57 | 0.2888 | 0.6757 |
| BQC-Dec matched confidence | 16.54 | +2.51 | 0.2424 | 0.6560 |
| **BQC-Dec quality ranking** | **17.73** | **+3.70** | **0.3900** | **0.7185** |

The quality score contributes a controlled **+1.19 R1@0.7 points** over
matched confidence ranking. All five seed-level deltas are positive.

## Research Status and Important Caveat

> [!WARNING]
> The checkpoints behind the table above were trained before an IoU-target
> gradient issue was identified. The historical implementation constructs
> `q_targets` from predicted spans without explicitly detaching them. In
> PyTorch, slice assignment retains a `CopySlices` gradient path, so the IoU
> regression term can also backpropagate through its target into predicted
> spans.

The repository separates the two versions explicitly:

```text
src/legacy_non_detached/   exact code behind the audited historical result
src/bqc/                   corrected target-detached implementation
```

The corrected implementation contains:

```python
q_targets = q_targets.detach()
```

The corrected version has **not yet been rerun**, so the historical 17.73
result must not be attributed to pure quality estimation until the controlled
rerun is complete. Full details are recorded in
[Code Audit Findings](docs/CODE_AUDIT.md).

## Repository Layout

```text
.
|-- src/
|   |-- baseline/                 original QD-DETR host file
|   |-- legacy_non_detached/      exact historical BQC implementation
|   `-- bqc/                      corrected detached-target implementation
|-- patches/
|   |-- qd_detr_bqc_legacy.patch
|   `-- qd_detr_bqc_detached.patch
|-- scripts/legacy_reproduction/  original server training/evaluation entries
|-- tools/audit_parameters.py     checkpoint parameter-delta audit
|-- audits/                       machine-readable audit output
`-- docs/                         architecture, experiment map, code audit
```

## Using the Code

The host implementation follows the DCASE AMR baseline built from QD-DETR.
Dataset annotations, extracted audio/text features, and checkpoints are not
redistributed.

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Apply the corrected BQC source

Either copy the corrected host file into a compatible QD-DETR AMR checkout:

```bash
cp src/bqc/qd_detr.py /path/to/amr_host/src/qd_detr.py
```

or inspect/apply the minimal patch:

```bash
git apply patches/qd_detr_bqc_detached.patch
```

### 3. Audit a checkpoint

```bash
python tools/audit_parameters.py \
  --base /path/to/host_checkpoint.pth \
  --bqc /path/to/bqc_checkpoint.pth \
  --output parameter_audit.json
```

The original experiment wrappers are preserved under
`scripts/legacy_reproduction/`. They intentionally retain the original server
paths for exact provenance and require path adaptation before use elsewhere.

## What the Method Does and Does Not Do

**BQC-Dec does:**

- estimate quality for candidates already produced by the host;
- train the ranking interface with IoU-aware and query-local supervision;
- improve selection within the available top-K candidate pool.

**BQC-Dec does not:**

- recover a moment missing from the candidate pool;
- use ground-truth windows during inference;
- directly refine start/end boundaries at test time;
- establish cross-host generalization.

## Reproducibility Notes

- [Experiment Map](docs/EXPERIMENT_MAP.md)
- [Architecture Notes](docs/ARCHITECTURE.md)
- [Code Audit Findings](docs/CODE_AUDIT.md)
- [Third-Party Notices](THIRD_PARTY_NOTICES.md)

Large artifacts remain outside Git. Checkpoints should never be committed to
this repository.

## References

- N. Carion et al., "End-to-End Object Detection with Transformers," ECCV,
  2020. [Paper](https://arxiv.org/abs/2005.12872)
- W. Moon et al., "Query-Dependent Video Representation for Moment Retrieval
  and Highlight Detection," CVPR, 2023.
  [Paper](https://arxiv.org/abs/2303.13874) |
  [Code](https://github.com/wjun0830/QD-DETR)
- H. Munakata et al., "Language-based Audio Moment Retrieval," 2024/2025.
  [Paper](https://arxiv.org/abs/2409.15672)

## Acknowledgements

This repository builds on DETR, Moment-DETR, QD-DETR, and the DCASE Audio
Moment Retrieval baseline. Their open-source implementations and task design
made the present analysis possible.
