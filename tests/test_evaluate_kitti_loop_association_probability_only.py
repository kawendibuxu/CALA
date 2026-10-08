import unittest

import pandas as pd

from evaluate_kitti_loop_association_probability_only import build_probability_only_selections


RULES = {
    "track_confidence_accept_min": 0.2,
    "visibility_accept_min": 0.2,
    "median_3d_residual_accept_max": 0.2,
}


class ProbabilityOnlyAblationTests(unittest.TestCase):
    def test_high_probability_without_geometry_is_selected(self):
        result = pd.DataFrame(
            [
                {
                    "query_dataset_index": 10,
                    "query_center_idx": 12,
                    "candidate_dataset_index": 1,
                    "candidate_center_idx": 3,
                    "candidate_rank": 1,
                    "pair_row_index": 0,
                    "label": "ignore",
                    "state": "uncertain",
                    "gt_distance_m": 8.0,
                    "calibrated_probability": 0.96,
                    "raw_logit": 14.0,
                    "mean_track_confidence": 0.01,
                    "mean_visibility": 0.01,
                    "median_3d_residual": 1.0,
                }
            ]
        )
        selected = build_probability_only_selections(result, 0.947, 5, RULES).iloc[0]
        self.assertEqual(selected.association_state, "selected")
        self.assertEqual(selected.selected_label, "ignore")
        self.assertFalse(bool(selected.passed_frozen_geometry))

    def test_geometry_pass_below_probability_abstains(self):
        result = pd.DataFrame(
            [
                {
                    "query_dataset_index": 10,
                    "query_center_idx": 12,
                    "candidate_dataset_index": 1,
                    "candidate_center_idx": 3,
                    "candidate_rank": 1,
                    "pair_row_index": 0,
                    "label": "positive",
                    "state": "uncertain",
                    "gt_distance_m": 1.0,
                    "calibrated_probability": 0.50,
                    "raw_logit": 4.0,
                    "mean_track_confidence": 0.9,
                    "mean_visibility": 0.9,
                    "median_3d_residual": 0.05,
                }
            ]
        )
        selected = build_probability_only_selections(result, 0.947, 5, RULES).iloc[0]
        self.assertEqual(selected.association_state, "abstain")

    def test_selection_uses_raw_logit_not_rank(self):
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
                    "state": "uncertain",
                    "gt_distance_m": 40.0,
                    "calibrated_probability": 0.95,
                    "raw_logit": 13.2,
                    "mean_track_confidence": 0.9,
                    "mean_visibility": 0.9,
                    "median_3d_residual": 0.05,
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
                    "calibrated_probability": 0.96,
                    "raw_logit": 16.0,
                    "mean_track_confidence": 0.01,
                    "mean_visibility": 0.01,
                    "median_3d_residual": 1.0,
                },
            ]
        )
        selected = build_probability_only_selections(result, 0.947, 5, RULES).iloc[0]
        self.assertEqual(selected.selected_label, "positive")
        self.assertEqual(int(selected.candidate_rank), 4)
        self.assertFalse(bool(selected.passed_frozen_geometry))


if __name__ == "__main__":
    unittest.main()
