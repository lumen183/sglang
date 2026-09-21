import sys
import unittest
from pathlib import Path

# Keep this pure-policy test independent of SGLang's torch-backed package
# initializer. The module under test has no runtime dependency on torch.
sys.path.insert(0, str(Path(__file__).parents[4] / "python/sglang/srt/managers"))

from hisparse_hybrid_policy import HybridHiSparsePolicy


class TestHybridHiSparsePolicy(unittest.TestCase):
    def test_watermark_uses_hot_cost_floor(self):
        policy = HybridHiSparsePolicy(total_blocks=100, hot_cost_blocks=15)
        self.assertEqual(policy.transition_watermark, 15)
        self.assertTrue(policy.is_under_pressure(14))
        self.assertFalse(policy.is_under_pressure(15))

    def test_watermark_uses_ten_percent_pool_floor(self):
        policy = HybridHiSparsePolicy(total_blocks=1000, hot_cost_blocks=4)
        self.assertEqual(policy.transition_watermark, 100)
        self.assertEqual(policy.reclaim_target(80, 12), 32)

    def test_reclaim_target_does_not_reclaim_without_pressure(self):
        policy = HybridHiSparsePolicy(total_blocks=1000, hot_cost_blocks=4)
        self.assertEqual(policy.reclaim_target(120, 12), 12)


if __name__ == "__main__":
    unittest.main()
