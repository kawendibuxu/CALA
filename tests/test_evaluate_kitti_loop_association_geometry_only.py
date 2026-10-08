import unittest

import pandas as pd

from evaluate_kitti_loop_association_geometry_only import (
    build_geometry_only_selections,
    geometry_mask,
    select_by_descriptor_rank,
)


RULES = {
    "track_confidence_accept_min": 0.1,
    "visibility_accept_min": 0.2,
    "median_3d_residual_accept_max": 0.3,
}


class GeometryOnlyAblationTests(unittest.TestCase):
    def test_geometry_mask_uses_all_three_gates(self):
        frame = pd.DataFrame(
            [
                {
                    "mean_track_confidence": 0.2,
                    "mean_visibility": 0.3,
                    "median_3d_residual": 0.1,
                },
                {
                    "mean_track_confidence": 0.05,
                    "mean_visibility": 0.3,
                    "median_3d_residual": 0.1,
                },
                {
                    "mean_track_confidence": 0.2,
                    "mean_visibility": 0.1,
                    "median_3d_residual": 0.1,
                },
                {
                    "mean_track_confidence": 0.2,
                    "mean_visibility": 0.3,
                    "median_3d_residual": 0.4,
                },
            ]
        )
        self.assertEqual(geometry_mask(frame, RULES).tolist(), [True, False, False, False])

    def test_select_by_rank_ignores_would_be_logit_order(self):
        candidates = pd.DataFrame(
            [
                {
                    "candidate_rank": 4,
                    "candidate_center_idx": 10,
                    "pair_row_index": 1,
                    "label": "positive",
                },
                {
                    "candidate_rank": 2,
                    "candidate_center_idx": 20,
                    "pair_row_index": 0,
                    "label": "ignore",
                },
            ]
        )
        selected = select_by_descriptor_rank(candidates)
        self.assertEqual(int(selected.candidate_rank), 2)
        self.assertEqual(selected.label, "ignore")

    def test_query_without_geometry_passers_abstains(self):
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
                    "mean_track_confidence": 0.01,
                    "mean_visibility": 0.01,
                    "median_3d_residual": 1.0,
                },
                {
                    "query_dataset_index": 11,
                    "query_center_idx": 13,
                    "candidate_dataset_index": 2,
                    "candidate_center_idx": 4,
                    "candidate_rank": 3,
                    "pair_row_index": 1,
                    "label": "negative",
                    "state": "reject",
                    "gt_distance_m": 40.0,
                    "mean_track_confidence": 0.5,
                    "mean_visibility": 0.5,
                    "median_3d_residual": 0.1,
                },
            ]
        )
        selections = build_geometry_only_selections(result, RULES).set_index("query_dataset_index")
        self.assertEqual(selections.loc[10, "association_state"], "abstain")
        self.assertEqual(selections.loc[11, "association_state"], "selected")
        self.assertEqual(selections.loc[11, "selected_label"], "negative")
        self.assertEqual(int(selections.loc[11, "candidate_rank"]), 3)


if __name__ == "__main__":
    unittest.main()
