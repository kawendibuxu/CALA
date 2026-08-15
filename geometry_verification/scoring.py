from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TrackRansacScore:
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    mean_sampson_error: float
    mean_track_confidence: float
    mean_visibility: float
    accepted: bool


def score_tracks_with_fundamental_ransac(
    points_a,
    points_b,
    confidence=None,
    visibility=None,
    ransac_reproj_threshold=1.5,
    min_matches=16,
    min_inlier_ratio=0.25,
):
    points_a = _to_numpy(points_a).astype(np.float32)
    points_b = _to_numpy(points_b).astype(np.float32)

    valid = np.isfinite(points_a).all(axis=1) & np.isfinite(points_b).all(axis=1)
    if confidence is not None:
        confidence_np = _to_numpy(confidence).reshape(-1)
        valid &= np.isfinite(confidence_np)
    else:
        confidence_np = np.ones(len(valid), dtype=np.float32)

    if visibility is not None:
        visibility_np = _to_numpy(visibility).reshape(-1)
        valid &= np.isfinite(visibility_np)
    else:
        visibility_np = np.ones(len(valid), dtype=np.float32)

    points_a = points_a[valid]
    points_b = points_b[valid]
    confidence_np = confidence_np[valid]
    visibility_np = visibility_np[valid]

    if len(points_a) < min_matches:
        return TrackRansacScore(
            num_matches=int(len(points_a)),
            num_inliers=0,
            inlier_ratio=0.0,
            mean_sampson_error=float("inf"),
            mean_track_confidence=float(confidence_np.mean()) if len(confidence_np) else 0.0,
            mean_visibility=float(visibility_np.mean()) if len(visibility_np) else 0.0,
            accepted=False,
        )

    fundamental, mask = cv2.findFundamentalMat(
        points_a,
        points_b,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=0.999,
        maxIters=10000,
    )

    if fundamental is None or mask is None:
        return TrackRansacScore(
            num_matches=int(len(points_a)),
            num_inliers=0,
            inlier_ratio=0.0,
            mean_sampson_error=float("inf"),
            mean_track_confidence=float(confidence_np.mean()),
            mean_visibility=float(visibility_np.mean()),
            accepted=False,
        )

    inlier_mask = mask.reshape(-1).astype(bool)
    num_inliers = int(inlier_mask.sum())
    inlier_ratio = num_inliers / max(1, len(points_a))
    mean_error = _mean_sampson_error(fundamental, points_a[inlier_mask], points_b[inlier_mask])

    return TrackRansacScore(
        num_matches=int(len(points_a)),
        num_inliers=num_inliers,
        inlier_ratio=float(inlier_ratio),
        mean_sampson_error=float(mean_error),
        mean_track_confidence=float(confidence_np[inlier_mask].mean()) if num_inliers else 0.0,
        mean_visibility=float(visibility_np[inlier_mask].mean()) if num_inliers else 0.0,
        accepted=bool(inlier_ratio >= min_inlier_ratio and num_inliers >= min_matches),
    )


def _mean_sampson_error(fundamental, points_a, points_b):
    if len(points_a) == 0:
        return float("inf")

    ones = np.ones((len(points_a), 1), dtype=np.float64)
    x1 = np.concatenate([points_a.astype(np.float64), ones], axis=1)
    x2 = np.concatenate([points_b.astype(np.float64), ones], axis=1)

    fx1 = fundamental @ x1.T
    ftx2 = fundamental.T @ x2.T
    x2tfx1 = np.sum(x2 * fx1.T, axis=1)
    denom = fx1[0] ** 2 + fx1[1] ** 2 + ftx2[0] ** 2 + ftx2[1] ** 2
    errors = x2tfx1**2 / np.maximum(denom, 1e-12)
    return float(errors.mean())


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
