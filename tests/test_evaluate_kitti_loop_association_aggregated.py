import unittest

import pandas as pd

from evaluate_kitti_loop_association_aggregated import (
    aggregate_selections,
    compare_with_frozen,
    neighbor_seeds,
    select_aggregated_candidate,
)
from evaluate_kitti_loop_association import build_selections


RULES = {
    "track_confidence_accept_min": 0.1,
    "visibility_accept_min": 0.16,
    "median_3d_residual_accept_max": 0.34,
}


def pair(query, pair_index, center, state, label, vis=0.4, track=0.4, residual=0.05, logit=10.0, rank=1):
    return {
        "query_dataset_index": query,
        "query_center_idx": query + 2,
        "candidate_dataset_index": max(center - 2, 0),
        "candidate_center_idx": center,
        "candidate_rank": rank,
        "pair_row_index": pair_index,
        "gt_distance_m": 1.0 if label == "positive" else 8.0 if label == "ignore" else 40.0,
        "calibrated_probability": 0.96 if state == "accept" else 0.5,
        "raw_logit": logit,
        "label": label,
        "state": state,
        "mean_track_confidence": track,
        "mean_visibility": vis,
        "median_3d_residual": residual,
    }


class TemporalAggregationTests(unittest.TestCase):
    def test_frozen_accept_is_copied_unchanged(self):
        result = pd.DataFrame(
            [
                pair(10, 0, 3, "accept", "positive", logit=20.0),
                pair(11, 1, 3, "uncertain", "positive", logit=8.0),
            ]
        )
        frozen = build_selections(result, cluster_gap=5)
        aggregated = aggregate_selections(result, frozen, window=3, cluster_gap=5, rules=RULES)
        compare_with_frozen(frozen, aggregated)
        kept = aggregated.loc[aggregated.aggregation_source.eq("frozen_accept")].iloc[0]
        self.assertEqual(int(kept.selected_pair_row_index), 0)
        self.assertEqual(kept.aggregation_source, "frozen_accept")

    def test_neighbor_within_window_promotes_same_cluster(self):
        result = pd.DataFrame(
            [
                pair(10, 0, 3, "accept", "positive", logit=20.0),
                pair(13, 1, 4, "uncertain", "positive", vis=0.3, track=0.3, residual=0.05, logit=8.0),
            ]
        )
        frozen = build_selections(result, cluster_gap=5)
        aggregated = aggregate_selections(result, frozen, window=3, cluster_gap=5, rules=RULES)
        promoted = aggregated.set_index("query_dataset_index").loc[13]
        self.assertEqual(promoted.association_state, "selected")
        self.assertEqual(promoted.aggregation_source, "temporal_aggregation")
        self.assertEqual(promoted.selected_label, "positive")
        self.assertEqual(int(promoted.seed_query_dataset_index), 10)

    def test_neighbor_outside_window_does_not_promote(self):
        result = pd.DataFrame(
            [
                pair(10, 0, 3, "accept", "positive", logit=20.0),
                pair(14, 1, 4, "uncertain", "positive", logit=8.0),
            ]
        )
        frozen = build_selections(result, cluster_gap=5)
        aggregated = aggregate_selections(result, frozen, window=3, cluster_gap=5, rules=RULES)
        self.assertEqual(
            aggregated.set_index("query_dataset_index").loc[14, "association_state"],
            "abstain",
        )

    def test_reject_and_failed_geometry_are_not_promoted(self):
        result = pd.DataFrame(
            [
                pair(10, 0, 3, "accept", "positive", logit=20.0),
                pair(11, 1, 3, "reject", "positive", vis=0.5, track=0.5, residual=0.05, logit=8.0),
                pair(12, 2, 3, "uncertain", "positive", vis=0.05, track=0.4, residual=0.05, logit=8.0),
            ]
        )
        frozen = build_selections(result, cluster_gap=5)
        aggregated = aggregate_selections(result, frozen, window=3, cluster_gap=5, rules=RULES)
        states = aggregated.set_index("query_dataset_index")
        self.assertEqual(states.loc[11, "association_state"], "abstain")
        self.assertEqual(states.loc[12, "association_state"], "abstain")

    def test_different_cluster_is_not_promoted(self):
        result = pd.DataFrame(
            [
                pair(10, 0, 3, "accept", "positive", logit=20.0),
                pair(11, 1, 40, "uncertain", "positive", logit=8.0),
            ]
        )
        frozen = build_selections(result, cluster_gap=5)
        aggregated = aggregate_selections(result, frozen, window=3, cluster_gap=5, rules=RULES)
        self.assertEqual(
            aggregated.set_index("query_dataset_index").loc[11, "association_state"],
            "abstain",
        )

    def test_ignore_promotion_is_allowed_but_counted_separately(self):
        result = pd.DataFrame(
            [
                pair(10, 0, 3, "accept", "positive", logit=20.0),
                pair(11, 1, 4, "uncertain", "ignore", vis=0.3, track=0.3, residual=0.05, logit=7.0),
            ]
        )
        frozen = build_selections(result, cluster_gap=5)
        aggregated = aggregate_selections(result, frozen, window=3, cluster_gap=5, rules=RULES)
        comparison = compare_with_frozen(frozen, aggregated)
        self.assertEqual(comparison["promoted_ignore"], 1)
        self.assertEqual(comparison["promoted_negative"], 0)

    def test_neighbor_seed_window_bounds(self):
        seeds = pd.DataFrame(
            {
                "query_dataset_index": [7, 10, 13, 14],
                "candidate_center_idx": [1, 1, 1, 1],
            }
        )
        nearby = neighbor_seeds(seeds, query_index=10, window=3)
        self.assertEqual(nearby.query_dataset_index.tolist(), [7, 13])


if __name__ == "__main__":
    unittest.main()
