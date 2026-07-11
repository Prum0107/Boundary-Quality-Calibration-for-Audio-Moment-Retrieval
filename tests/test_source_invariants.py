import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SourceInvariantTests(unittest.TestCase):
    def test_corrected_source_detaches_iou_target(self):
        source = (ROOT / "src/bqc/qd_detr.py").read_text(encoding="utf-8")
        self.assertIn("q_targets = q_targets.detach()", source)

    def test_legacy_source_preserves_historical_behavior(self):
        source = (ROOT / "src/legacy_non_detached/qd_detr.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("q_targets = q_targets.detach()", source)

    def test_freeze_policy_targets_decoder_not_moment_queries(self):
        source = (
            ROOT / "scripts/legacy_reproduction/train_bqc_dec.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"transformer.decoder" in name', source)
        self.assertIn('"quality_head" in name', source)
        self.assertNotIn('"query_embed" in name', source)


if __name__ == "__main__":
    unittest.main()
