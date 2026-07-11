# Code Audit Findings

Date: 2026-07-11

## 1. Parameter Scope: Confirmed

The seed-2026 checkpoint was compared directly with the frozen host checkpoint.

| Group | Policy | Parameters | Changed tensors | Delta L2 |
|---|---:|---:|---:|---:|
| Moment queries (`query_embed`) | frozen | 20 | 0 / 1 | 0 |
| Transformer decoder | trainable | 3,160,837 | 89 / 92 | 1.48305 |
| Quality head | newly added | 66,306 | not in base | n/a |
| Span/class heads | frozen | 132,612 | 0 / 8 | 0 |
| Encoder | frozen | 1,579,522 | 0 / 26 | 0 |
| Other host parameters | frozen | 2,250,246 | 0 / 51 | 0 |

Conclusion: the current checkpoint fine-tunes transformer decoder layers and
adds a quality head. The learned moment anchors/queries are not updated.

Machine-readable evidence:
[`audits/parameter_audit_seed2026.json`](../audits/parameter_audit_seed2026.json).

## 2. IoU Regression Target: Blocking Issue

The historical implementation behind the reported checkpoints constructs its
target by slice assignment from an IoU tensor that depends on `pred_spans`:

```python
q_targets[b] = iou_mat.max(dim=1)[0]
L_qreg = smooth_l1_loss(q_pred, q_targets)
```

PyTorch preserves a `CopySlices` gradient path through this assignment. A
server-side minimal reproduction confirmed:

```text
target_requires_grad True
target_grad_fn <CopySlices ...>
source_grad [-0.15, 0.15]
prediction_grad [0.15, -0.15]
```

Consequently, the historical regression loss is not a pure quality-estimation
loss. It also sends gradients through the target IoU into predicted spans and
the decoder. The classification target and list-wise target do not have the
same path because thresholding/no-grad blocks it.

The corrected implementation in `src/bqc/qd_detr.py` applies:

```python
q_targets = q_targets.detach()
```

The exact historical implementation is retained in
`src/legacy_non_detached/qd_detr.py`. A controlled rerun of the corrected code
is required before the pure quality-estimation mechanism claim is retained.
Existing checkpoints remain valid historical artifacts but are labelled
`legacy_non_detached` until that rerun is complete.

## 3. Claims Allowed Before Rerun

Allowed:

- decoder parameters changed while moment queries and prediction heads stayed
  fixed;
- the trained system improved the audited ranking metrics;
- the quality objective as implemented influenced decoder optimization.

Not yet allowed:

- the gain comes from pure boundary-quality estimation;
- decoder features learned only to predict a fixed IoU target;
- the regression loss is independent of span optimization.
