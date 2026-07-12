# Detached-Target Mechanism Ablation

Date: 2026-07-11  
Host: QD-DETR initialized from the same Clotho-pretrained, CASTELLA-finetuned checkpoint  
Evaluation: CASTELLA development-testing, 1,347 queries
Seeds: 2026, 2027, 2028, 2029, 2030

## Purpose

The experiment separates four possible explanations for the previously
reported BQC-Dec gain:

1. a quality head can learn from frozen decoder slots;
2. moment-anchor adaptation changes the candidate pool;
3. ordinary decoder fine-tuning under the host loss explains the gain;
4. correct detached IoU supervision contributes beyond decoder adaptation.

Every IoU target is computed under `torch.no_grad()` and explicitly detached.
Model selection uses development-validation only. Quality variants are ranked by `qcls`, while
the host-only variant is ranked by foreground confidence. No score rule is
selected on development-testing.

## Primary Five Variants

| Variant | Trainable parameters | Training loss | R1@0.7 mean +/- std | Delta vs frozen host | Spearman | Pairwise accuracy |
|---|---|---|---:|---:|---:|---:|
| Quality head only | Quality head | Detached quality | 16.27 +/- 0.06 | +1.95 | 0.367 | 0.696 |
| Anchors + quality head | Moment anchors, quality head | Detached quality | 16.38 +/- 0.12 | +2.05 | 0.369 | 0.696 |
| Decoder + host loss | Decoder | Original host | 16.20 +/- 0.31 | +1.87 | 0.237 | 0.643 |
| Decoder + detached quality loss | Decoder, quality head | Detached quality only | 10.32 +/- 1.88 | -4.01 | 0.366 | 0.695 |
| Decoder + shuffled quality labels | Decoder, quality head | Shuffled detached quality only | 6.52 +/- 1.53 | -7.81 | 0.283 | 0.633 |

Frozen-host R1@0.7 is 14.33%, Spearman is 0.239, and pairwise accuracy is
0.645 under the same evaluator.

## Necessary Joint-Loss Controls

Quality-only decoder adaptation changes slot representations without preserving
the host localization objective. Two additional controls retain the original
host losses and isolate whether correct IoU labels add useful information.

| Variant | R1@0.7 mean +/- std | Delta vs frozen host | Spearman | Pairwise accuracy |
|---|---:|---:|---:|---:|
| Decoder + host + detached quality | **17.76 +/- 0.27** | **+3.43** | **0.392** | **0.699** |
| Decoder + host + shuffled quality | 13.63 +/- 1.70 | -0.70 | 0.313 | 0.649 |

The five per-seed R1@0.7 values for the corrected joint objective are 17.37,
17.97, 17.89, 17.59, and 17.97. The paired improvement over decoder + host loss
is +1.26, +1.27, +2.00, +1.55, and +1.71 points, respectively, for a mean
quality-supervision contribution of **+1.56 points**.

## Gradient Audit

All 35 runs recorded `target_requires_grad=false`. Seed-2026 smoke checks
showed the expected non-zero gradient groups:

| Variant | Anchors | Decoder | Quality head | Other |
|---|---:|---:|---:|---:|
| Quality head only | 0 | 0 | 5.726 | 0 |
| Anchors + quality head | 0.017 | 0 | 5.726 | 0 |
| Decoder + host | 0 | 24.593 | 0 | 0 |
| Decoder + detached quality | 0 | 3.219 | 5.726 | 0 |
| Decoder + shuffled quality | 0 | 2.424 | 4.671 | 0 |
| Decoder + host + detached quality | 0 | 25.806 | 5.726 | 0 |
| Decoder + host + shuffled quality | 0 | 25.553 | 4.671 | 0 |

The values are sums of per-tensor gradient norms and are used only to verify
gradient routing, not to compare optimization magnitude across modules.

## Interpretation

1. Frozen decoder slots already contain enough information for a lightweight
   quality head to improve candidate selection by about two R1@0.7 points.
2. Updating moment anchors adds only 0.10 points over quality-head-only training;
   anchor adaptation is not the main mechanism.
3. Continuing decoder training with the original host loss gives +1.87 points,
   so decoder adaptation is a substantial matched control and must be reported.
4. Detached quality loss without host loss improves score-IoU alignment but
   damages localization. Spearman alone cannot establish AMR improvement.
5. Correct detached quality supervision combined with host loss reaches 17.76%,
   1.56 points above host-only decoder adaptation for the same five seeds.
6. Shuffling quality labels collapses the joint result to 13.63%. The gain is
   attributable to the candidate-specific IoU supervision, not merely an extra
   loss term or generic regularization.

## Recommended Claim

Under a corrected detached-target implementation, boundary-quality supervision
provides a controlled +1.56 R1@0.7 points over matched decoder adaptation. The
full corrected system improves the frozen host from 14.33% to 17.76% on average
across five seeds. Moment-anchor adaptation is unnecessary for the main gain.

Server artifacts are stored in:

```text
/root/autodl-tmp/bqc_mechanism_ablation_20260711/
```
