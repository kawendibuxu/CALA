import unittest

import pandas as pd

from evaluate_kitti_loop_association_ablations import (
    build_rank1_selections,
    build_score_rerank_selections,
)


class AssociationAblationTests(unittest.TestCase):
    def test_rank1_selects_only_rank_one(self):
        pairs = pd.DataFrame(
            [
                {
                    "query_dataset_index": 10,
                    "query_center_idx": 12,
                    "candidate_dataset_index": 1,
                    "candidate_center_idx": 3,
                    "candidate_rank": 1,
                    "label": "ignore",
                    "gt_distance_m": 8.0,
                    "descriptor_similarity": 0.9,
                },
                {
                    "query_dataset_index": 10,
                    "query_center_idx": 12,
                    "candidate_dataset_index": 2,
                    "candidate_center_idx": 4,
                    "candidate_rank": 2,
                    "label": "positive",
                    "gt_distance_m": 1.0,
                    "descriptor_similarity": 0.8,
                },
            ]
        )
        selected = build_rank1_selections(pairs).iloc[0]
        self.assertEqual(int(selected.candidate_rank), 1)
        self.assertEqual(selected.selected_label, "ignore")
        self.assertTrue(bool(selected.retrieved_positive))

    def test_rank1_requires_exactly_one_rank_one_row(self):
        pairs = pd.DataFrame(
            [
                {
                    "query_dataset_index": 10,
                    "query_center_idx": 12,
                    "candidate_dataset_index": 1,
                    "candidate_center_idx": 3,
                    "candidate_rank": 2,
                    "label": "positive",
                    "gt_distance_m": 1.0,
                    "descriptor_similarity": 0.8,
                }
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "no historical rank-1"):
            build_rank1_selections(pairs)

    def test_score_rerank_ignores_accept_gate(self):
        result = pd.DataFrame(
            [
                {
                    "query_dataset_index": 10,
                    "query_center_idx": 12,
                    "candidate_dataset_index": 1,
                    "candidate_center_idx": 3,
                    "candidate_rank": 1,
                    "pair_row_index": 0,
                    "label": "negative",
                    "state": "accept",
                    "gt_distance_m": 40.0,
                    "raw_logit": 5.0,
                },
                {
                    "query_dataset_index": 10,
                    "query_center_idx": 12,
                    "candidate_dataset_index": 2,
                    "candidate_center_idx": 20,
                    "candidate_rank": 4,
                    "pair_row_index": 1,
                    "label": "positive",
                    "state": "uncertain",
                    "gt_distance_m": 1.0,
                    "raw_logit": 12.0,
                },
            ]
        )
        selected = build_score_rerank_selections(result, cluster_gap=5).iloc[0]
        self.assertEqual(selected.selected_label, "positive")
        self.assertEqual(int(selected.candidate_rank), 4)
        self.assertEqual(selected.association_state, "selected")
        self.assertEqual(selected.selected_three_way_state, "uncertain")
        self.assertEqual(int(selected.selected_pair_row_index), 1)


if __name__ == "__main__":
    unittest.main()
