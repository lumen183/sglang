#!/usr/bin/env python3

import unittest

from assert_coverage import analyze


class TestCoverageAnalysis(unittest.TestCase):
    def test_resident_path(self):
        result = analyze(
            "hybrid-resident",
            "HiSparse hybrid event=resident_admit\n"
            "HiSparse hybrid event=resident_topk\n",
        )
        self.assertTrue(result["passed"])

    def test_resident_rejects_demotion(self):
        result = analyze(
            "hybrid-resident",
            "HiSparse hybrid event=resident_admit\n"
            "HiSparse hybrid event=resident_topk\n"
            "HiSparse hybrid event=demote freed_c4_slots=64\n",
        )
        self.assertFalse(result["passed"])

    def test_evict_path(self):
        result = analyze(
            "hybrid-evict",
            "HiSparse hybrid event=resident_admit\n"
            "HiSparse hybrid event=mirror_complete\n"
            "HiSparse hybrid event=demote freed_c4_slots=64\n"
            "HiSparse hybrid event=host_topk\n",
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["freed_c4_slots"], 64)

    def test_evict_requires_order(self):
        result = analyze(
            "hybrid-evict",
            "HiSparse hybrid event=resident_admit\n"
            "HiSparse hybrid event=demote freed_c4_slots=64\n"
            "HiSparse hybrid event=mirror_complete\n"
            "HiSparse hybrid event=host_topk\n",
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
