import unittest

import numpy as np
import pandas as pd

from evaluate_kitti_loop_association import (
    add_temporal_clusters,
    association_precision_recall_curve,
    build_selections,
    historical_positive_query_ids,
    select_one_candidate,
)


class LoopAssociationEvaluationTests(unittest.TestCase):
    def test_temporal_clusters_use_center_frame_gap(self):
        candidates = pd.DataFrame(
            {
                "candidate_center_idx": [20, 1, 14, 3, 10],
                "candidate_rank": [5, 1, 4, 2, 3],
                "pair_row_index": [4, 0, 3, 1, 2],
            }
        )
        clustered = add_temporal_clusters(candidates, cluster_gap=5)
        self.assertEqual(
            clustered.sort_values("candidate_center_idx").temporal_cluster_id.tolist(),
            [0, 0, 1, 1, 2],
        )

    def test_selection_uses_raw_logit_then_deterministic_ties(self):
        candidates = pd.DataFrame(
            {
                "candidate_center_idx": [100, 101, 300],
                "candidate_rank": [3, 2, 1],
                "pair_row_index": [0, 1, 2],
                "raw_logit": [4.0, 4.0, 3.0],
                "label": ["positive", "positive", "negative"],
            }
        )
        selected = select_one_candidate(candidates, cluster_gap=5)
        self.assertEqual(int(selected.pair_row_index), 1)
        self.assertEqual(int(selected.cluster_count), 2)
        self.assertEqual(int(selected.selected_cluster_size), 2)

    def test_build_selections_emits_one_candidate_or_abstains(self):
        result = pd.DataFrame(
            [
                {
                    "query_dataset_index": 10,
                    "query_center_idx": 12,
                    "candidate_dataset_index": 1,
                    "candidate_center_idx": 3,
                    "candidate_rank": 1,
                    "pair_row_index": 0,
                    "gt_distance_m": 1.0,
                    "calibrated_probability": 0.96,
                    "raw_logit": 15.0,
                    "label": "positive",
                    "state": "accept",
                },
                {
                    "query_dataset_index": 11,
                    "query_center_idx": 13,
                    "candidate_dataset_index": 1,
                    "candidate_center_idx": 3,
                    "candidate_rank": 1,
                    "pair_row_index": 1,
                    "gt_distance_m": 30.0,
                    "calibrated_probability": 0.001,
                    "raw_logit": -5.0,
                    "label": "negative",
                    "state": "reject",
                },
            ]
        )
        selections = build_selections(result, cluster_gap=5).set_index("query_dataset_index")
        self.assertEqual(selections.loc[10, "association_state"], "selected")
        self.assertEqual(selections.loc[10, "selected_label"], "positive")
        self.assertEqual(selections.loc[11, "association_state"], "abstain")

    def test_historical_positive_query_scan(self):
        centers = np.asarray([0, 100, 200])
        translations = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        positives = historical_positive_query_ids(
            centers,
            translations,
            min_temporal_gap=100,
            positive_radius=5.0,
            distance_axes=(0, 2),
        )
        self.assertEqual(positives, {2})

    def test_precision_recall_curve_counts_wrong_associations(self):
        predictions = pd.DataFrame(
            {
                "query_dataset_index": [1, 2, 3],
                "raw_logit": [3.0, 2.0, 1.0],
                "label": ["positive", "ignore", "positive"],
            }
        )
        curve, average_precision, recall_at_100_precision = association_precision_recall_curve(
            predictions,
            historical_positive_queries=2,
        )
        self.assertEqual(curve.true_positives.tolist(), [1, 1, 2])
        self.assertEqual(curve.false_positives.tolist(), [0, 1, 1])
        self.assertAlmostEqual(average_precision, (1.0 + 2.0 / 3.0) / 2.0)
        self.assertAlmostEqual(recall_at_100_precision, 0.5)


if __name__ == "__main__":
    unittest.main()
