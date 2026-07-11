# Legacy Reproduction Scripts

The scripts in this directory are exact copies of the server-side experiment
entries used to produce and audit the historical results. They intentionally
retain absolute paths from that environment.

They are preserved for provenance, not presented as portable entry points.
Before running them elsewhere:

1. replace the `REPO`, `OUT_DIR`, and checkpoint paths;
2. use the corrected source from `src/bqc/qd_detr.py`;
3. write new outputs under a distinct detached-target experiment directory;
4. do not overwrite the historical checkpoints.

