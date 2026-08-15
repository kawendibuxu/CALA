"""Explicit 3D cross-token verification using frozen VGGT predictions."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .image_io import make_grid_query_points
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


@dataclass
class CrossToken3DScore:
    num_query_tokens: int
    num_cross_frame_matches: int
    num_3d_inliers: int
    weighted_3d_inlier_ratio: float
    median_3d_residual: float
    p90_3d_residual: float
    mean_match_similarity: float
    mean_track_confidence: float
    mean_visibility: float
    mean_point_confidence: float
    mean_depth_confidence: float
    frame_support: float
    temporal_consistency: float
    camera_rotation_consistency_deg: float
    camera_translation_consistency: float


def _sample_map(values, points):
    """Bilinearly sample [F, H, W, C] values at [F, N, 2] pixel points."""
    frames, height, width, channels = values.shape
    if points.shape[0] != frames:
        raise ValueError("Point and map frame dimensions must match.")
    normalized = points.clone()
    normalized[..., 0] = normalized[..., 0] / max(width - 1, 1) * 2 - 1
    normalized[..., 1] = normalized[..., 1] / max(height - 1, 1) * 2 - 1
    grid = normalized.unsqueeze(2)
    sampled = F.grid_sample(
        values.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return sampled.squeeze(-1).permute(0, 2, 1)


def _weighted_kabsch(source, target, weights):
    weights = weights / np.maximum(weights.sum(), 1e-12)
    source_center = (source * weights[:, None]).sum(axis=0)
    target_center = (target * weights[:, None]).sum(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (source_zero * weights[:, None]).T @ target_zero
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _rotation_angle_deg(rotation):
    cosine = np.clip((np.trace(rotation) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def _ransac_3d(source, target, weights, threshold_m, iterations, seed):
    if len(source) < 3:
        return None, None, np.zeros(len(source), dtype=bool), np.full(len(source), np.inf)
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(len(source), dtype=bool)
    best_score = -1.0
    for _ in range(iterations):
        sample = rng.choice(len(source), size=3, replace=False, p=weights / weights.sum())
        try:
            rotation, translation = _weighted_kabsch(source[sample], target[sample], weights[sample])
        except np.linalg.LinAlgError:
            continue
        residuals = np.linalg.norm((source @ rotation.T) + translation - target, axis=1)
        inliers = residuals <= threshold_m
        score = weights[inliers].sum()
        if score > best_score:
            best_mask = inliers
            best_score = score
    if best_mask.sum() < 3:
        return None, None, best_mask, np.full(len(source), np.inf)
    rotation, translation = _weighted_kabsch(source[best_mask], target[best_mask], weights[best_mask])
    residuals = np.linalg.norm((source @ rotation.T) + translation - target, axis=1)
    return rotation, translation, residuals <= threshold_m, residuals


def _camera_consistency(pose_encoding, sequence_length):
    extrinsics, _ = pose_encoding_to_extri_intri(
        pose_encoding.unsqueeze(0), build_intrinsics=False
    )
    extrinsics = extrinsics[0].detach().float().cpu().numpy()
    query = extrinsics[:sequence_length]
    candidate = extrinsics[sequence_length:]
    rotations = []
    translations = []
    for query_pose, candidate_pose in zip(query, candidate):
        query_rotation, query_translation = query_pose[:, :3], query_pose[:, 3]
        candidate_rotation, candidate_translation = candidate_pose[:, :3], candidate_pose[:, 3]
        rotation = candidate_rotation.T @ query_rotation
        translation = candidate_rotation.T @ (query_translation - candidate_translation)
        rotations.append(rotation)
        translations.append(translation)
    reference_rotation = rotations[0]
    rotation_degrees = [_rotation_angle_deg(rotation @ reference_rotation.T) for rotation in rotations]
    translations = np.asarray(translations)
    return float(np.median(rotation_degrees)), float(np.median(np.linalg.norm(translations - np.median(translations, axis=0), axis=1)))


def score_cross_token_3d(predictions, sequence_length, image_size, grid_rows, grid_cols,
                         ransac_threshold_m=0.5, ransac_iterations=256, seed=0):
    """Build center-frame tokens and verify them across all candidate frames.

    VGGT predicts every image jointly.  A fixed grid in query frame zero is tracked
    into candidate frames, where each 2D track is lifted from VGGT's point map.
    This preserves patch coordinates and candidate frame offsets for every 3D match.
    """
    required = {"world_points", "world_points_conf", "depth_conf", "track", "vis", "conf", "pose_enc"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"VGGT predictions are missing: {sorted(missing)}")
    height, width = image_size
    query_points = make_grid_query_points(image_size, grid_rows, grid_cols).to(predictions["track"].device)
    token_count = len(query_points)
    points = predictions["world_points"][0]
    point_conf = predictions["world_points_conf"][0]
    depth_conf = predictions["depth_conf"][0]
    tracks = predictions["track"][0]
    visibility = predictions["vis"][0]
    track_conf = predictions["conf"][0]

    source_points_2d = query_points.unsqueeze(0)
    source_3d = _sample_map(points[:1], source_points_2d)[0]
    source_point_conf = _sample_map(point_conf[:1].unsqueeze(-1), source_points_2d)[0, :, 0]
    source_depth_conf = _sample_map(depth_conf[:1].unsqueeze(-1), source_points_2d)[0, :, 0]

    candidate_points_2d = tracks[sequence_length:]
    candidate_3d = _sample_map(points[sequence_length:], candidate_points_2d)
    candidate_point_conf = _sample_map(point_conf[sequence_length:].unsqueeze(-1), candidate_points_2d)[..., 0]
    candidate_depth_conf = _sample_map(depth_conf[sequence_length:].unsqueeze(-1), candidate_points_2d)[..., 0]
    candidate_visibility = visibility[sequence_length:]
    candidate_track_conf = track_conf[sequence_length:]

    source_3d = source_3d.detach().float().cpu().numpy()
    candidate_3d = candidate_3d.detach().float().cpu().numpy()
    source_point_conf = source_point_conf.detach().float().cpu().numpy()
    source_depth_conf = source_depth_conf.detach().float().cpu().numpy()
    candidate_point_conf = candidate_point_conf.detach().float().cpu().numpy()
    candidate_depth_conf = candidate_depth_conf.detach().float().cpu().numpy()
    candidate_visibility = candidate_visibility.detach().float().cpu().numpy()
    candidate_track_conf = candidate_track_conf.detach().float().cpu().numpy()

    # A match is one source patch token observed in one candidate frame.
    source_repeated = np.broadcast_to(source_3d, candidate_3d.shape).reshape(-1, 3)
    target_flat = candidate_3d.reshape(-1, 3)
    vis_flat = candidate_visibility.reshape(-1)
    track_flat = candidate_track_conf.reshape(-1)
    point_flat = candidate_point_conf.reshape(-1)
    depth_flat = candidate_depth_conf.reshape(-1)
    source_point_flat = np.tile(source_point_conf, sequence_length)
    source_depth_flat = np.tile(source_depth_conf, sequence_length)
    frame_ids = np.repeat(np.arange(sequence_length), token_count)
    finite = np.isfinite(source_repeated).all(axis=1) & np.isfinite(target_flat).all(axis=1)
    valid = finite & (vis_flat > 0) & (track_flat > 0) & (point_flat > 0) & (depth_flat > 0)
    source_repeated, target_flat = source_repeated[valid], target_flat[valid]
    vis_flat, track_flat = vis_flat[valid], track_flat[valid]
    point_flat, depth_flat = point_flat[valid], depth_flat[valid]
    source_point_flat, source_depth_flat = source_point_flat[valid], source_depth_flat[valid]
    frame_ids = frame_ids[valid]
    weights = vis_flat * track_flat * point_flat * depth_flat * source_point_flat * source_depth_flat
    weights = np.maximum(weights, 1e-12)

    rotation, translation, inliers, residuals = _ransac_3d(
        source_repeated, target_flat, weights, ransac_threshold_m, ransac_iterations, seed
    )
    if rotation is None:
        inliers = np.zeros(len(source_repeated), dtype=bool)
        residuals = np.full(len(source_repeated), np.inf)
    finite_residuals = residuals[np.isfinite(residuals)]
    inlier_frames = np.unique(frame_ids[inliers])
    frame_support = len(inlier_frames) / max(sequence_length, 1)
    per_frame_ratio = []
    for frame_id in range(sequence_length):
        frame_mask = frame_ids == frame_id
        if frame_mask.any():
            per_frame_ratio.append(float(inliers[frame_mask].mean()))
    temporal_consistency = float(np.mean(per_frame_ratio)) if per_frame_ratio else 0.0
    camera_rotation, camera_translation = _camera_consistency(predictions["pose_enc"][0], sequence_length)
    return CrossToken3DScore(
        num_query_tokens=token_count,
        num_cross_frame_matches=len(source_repeated),
        num_3d_inliers=int(inliers.sum()),
        weighted_3d_inlier_ratio=float(weights[inliers].sum() / weights.sum()) if len(weights) else 0.0,
        median_3d_residual=float(np.median(finite_residuals)) if len(finite_residuals) else float("inf"),
        p90_3d_residual=float(np.quantile(finite_residuals, 0.9)) if len(finite_residuals) else float("inf"),
        mean_match_similarity=0.0,  # Track head does not expose token descriptors in the official API.
        mean_track_confidence=float(track_flat[inliers].mean()) if inliers.any() else 0.0,
        mean_visibility=float(vis_flat[inliers].mean()) if inliers.any() else 0.0,
        mean_point_confidence=float(point_flat[inliers].mean()) if inliers.any() else 0.0,
        mean_depth_confidence=float(depth_flat[inliers].mean()) if inliers.any() else 0.0,
        frame_support=float(frame_support),
        temporal_consistency=temporal_consistency,
        camera_rotation_consistency_deg=camera_rotation,
        camera_translation_consistency=camera_translation,
    )
