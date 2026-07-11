# Architecture Notes

## Host Forward Path

```text
audio/text features
  -> query-dependent encoder
  -> encoded temporal memory
  + query_embed.weight (K learned moment queries, shape K x 2)
  -> transformer decoder
  -> hs[layer, batch, slot, hidden]
  -> class_embed(hs): foreground confidence
  -> span_embed(hs) + reference: center/width
```

The host source names the learned moment anchors `query_embed`. They are
inputs to the transformer decoder, not the decoder itself.

## BQC Forward Path

```text
hs[-1]
  -> frozen class/span readouts
  -> added quality_head
       -> quality_reg
       -> quality_logit
```

Quality targets are derived on train data from each predicted candidate and
the annotated temporal windows:

```text
q_i = max_g IoU(candidate_i, ground_truth_g)
y_i = 1[q_i >= 0.7]
```

The three losses are continuous IoU regression, IoU@0.7 classification, and
within-query list-wise ordering.

## Source Versions

- `src/legacy_non_detached/qd_detr.py` is the exact implementation behind the
  audited historical checkpoints. Its IoU regression target retains a gradient
  path through predicted spans.
- `src/bqc/qd_detr.py` explicitly detaches the IoU target. It is the corrected
  implementation intended for the next controlled run.

Results from the corrected implementation are not reported yet. The distinction
is kept explicit so source provenance and mechanism claims cannot be conflated.

## Train/Test Boundary

Ground-truth windows are used only to construct training targets. During
evaluation, the host generates candidate windows and the quality head predicts
their scores without access to ground truth.

## Supported Claim

BQC-Dec improves quality-aware selection within the generated candidate pool.
It is not a new candidate generator and does not directly refine start/end
boundaries at inference.
