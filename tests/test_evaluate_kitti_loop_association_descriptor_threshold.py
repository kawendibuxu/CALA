import unittest

import pandas as pd

from evaluate_kitti_loop_association_descriptor_threshold import (
    apply_similarity_threshold,
    calibrate_similarity_threshold,
    rank1_rows,
)


class DescriptorThresholdAblationTests(unittest.TestCase):
    def test_calibrate_uses_lowest_threshold_meeting_strict_precision(self):
        rank1 = pd.DataFrame(
            [
                {"query_dataset_index": 1, "label": "positive", "descriptor_similarity": 0.95},
                {"query_dataset_index": 2, "label": "positive", "descriptor_similarity": 0.90},
                {"query_dataset_index": 3, "label": "ignore", "descriptor_similarity": 0.85},
                {"query_dataset_index": 4, "label": "negative", "descriptor_similarity": 0.80},
            ]
        )
        stats = calibrate_similarity_threshold(rank1, target_precision=0.99)
        self.assertAlmostEqual(stats["threshold"], 0.90)
        self.assertEqual(stats["correct_associations"], 2)
        self.assertEqual(stats["gray_ignore_associations"], 0)

    def test_calibrate_treats_ignore_as_incorrect(self):
        rank1 = pd.DataFrame(
            [
                {"query_dataset_index": 1, "label": "positive", "descriptor_similarity": 0.99},
                {"query_dataset_index": 2, "label": "ignore", "descriptor_similarity": 0.98},
            ]
        )
        stats = calibrate_similarity_threshold(rank1, target_precision=0.99)
        self.assertAlmostEqual(stats["threshold"], 0.99)
        self.assertEqual(stats["selected_queries"], 1)

    def test_apply_threshold_keeps_rank1_and_abstains_below(self):
        selections = pd.DataFrame(
            [
                {
                    "query_dataset_index": 10,
                    "query_center_idx": 12,
                    "retrieved_positive": True,
                    "selected_label": "positive",
                    "selected_pair_row_index": 0,
                    "candidate_dataset_index": 1,
                    "candidate_center_idx": 3,
                    "candidate_rank": 1,
                    "gt_distance_m": 1.0,
                    "score": 0.91,
                    "candidate_count": 1,
                    "cluster_count": 1,
                    "selected_cluster_size": 1,
                },
                {
                    "query_dataset_index": 11,
                    "query_center_idx": 13,
                    "retrieved_positive": False,
                    "selected_label": "negative",
                    "selected_pair_row_index": 1,
                    "candidate_dataset_index": 2,
                    "candidate_center_idx": 4,
                    "candidate_rank": 1,
                    "gt_distance_m": 40.0,
                    "score": 0.70,
                    "candidate_count": 1,
                    "cluster_count": 1,
                    "selected_cluster_size": 1,
                },
            ]
        )
        out = apply_similarity_threshold(selections, threshold=0.90).set_index("query_dataset_index")
        self.assertEqual(out.loc[10, "association_state"], "selected")
        self.assertEqual(out.loc[10, "selected_label"], "positive")
        self.assertEqual(int(out.loc[10, "candidate_rank"]), 1)
        self.assertEqual(out.loc[11, "association_state"], "abstain")
        self.assertEqual(out.loc[11, "selected_label"], "")

    def test_rank1_rows_reject_missing_or_duplicate(self):
        missing = pd.DataFrame(
            [
                {
                    "query_dataset_index": 1,
                    "candidate_rank": 2,
                    "label": "positive",
                    "descriptor_similarity": 0.9,
                }
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "rank-1"):
            rank1_rows(missing)


if __name__ == "__main__":
    unittest.main()
