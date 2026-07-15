# Experiment Map

## Main Evidence

| Question | Canonical evidence |
|---|---|
| What is the frozen host result? | paper claim audit and matched evaluator |
| Does the candidate pool contain better windows? | Oracle@10 audit |
| Does post-hoc reranking work? | grouped train-CV diagnostic |
| Does BQC-Dec improve top-1 selection? | five BQC-Dec checkpoints |
| What is the ranking-only contribution? | same-checkpoint matched confidence vs quality |
| Which parameters are updated? | `audit_parameters.py` and freeze policy |

## Historical Branches

- `bql_training_20260705`: initial quality-learning implementation.
- `bql_v2_20260705`: loss and feature variants.
- `bql_final_20260705`: structure and loss ablations.
- `bql_dec_br_20260705`: five BQC-Dec checkpoints and refinement work.
- `bql_predicted_tier_20260705`: deployable refinement gates.
- `varifocal_iou_conf_20260708`: direct IoU-aware confidence baseline.
- `bqc_joint_20260708`: joint objective; appendix-only negative control.
- `bqc_paper_audit`: final evaluator and numeric audit.

The `bql_*` strings above are immutable historical server directory names from
the project's earlier writing stage. They do not name a separate current
method. All active source code, losses, and public interfaces use BQC.
