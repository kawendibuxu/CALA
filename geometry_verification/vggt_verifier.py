from dataclasses import dataclass
from pathlib import Path

import torch

from vggt.models.vggt import VGGT

from .image_io import make_grid_query_points
from .scoring import TrackRansacScore, score_tracks_with_fundamental_ransac
from .cross_token_3d import CrossToken3DScore, score_cross_token_3d


@dataclass
class GeometryVerificationResult:
    score: TrackRansacScore
    raw_predictions: dict
    load_missing_keys: list[str]
    load_unexpected_keys: list[str]


@dataclass
class CrossToken3DVerificationResult:
    score: CrossToken3DScore
    load_missing_keys: list[str]
    load_unexpected_keys: list[str]


class VGGTGeometryVerifier:
    def __init__(
        self,
        checkpoint_path,
        device=None,
        image_size=(392, 518),
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = tuple(image_size)
        self.model = VGGT(
            enable_camera=enable_camera,
            enable_point=enable_point,
            enable_depth=enable_depth,
            enable_track=enable_track,
        )
        self.missing_keys, self.unexpected_keys = self._load_checkpoint(checkpoint_path)
        self.model.eval().to(self.device)

    def verify_pair(
        self,
        query_sequence,
        candidate_sequence,
        query_points=None,
        candidate_frame_offset=0,
        grid_rows=12,
        grid_cols=16,
        ransac_reproj_threshold=1.5,
    ):
        if query_sequence.ndim != 4 or candidate_sequence.ndim != 4:
            raise ValueError("Expected query and candidate sequences with shape [S, 3, H, W].")
        if query_sequence.shape != candidate_sequence.shape:
            raise ValueError(
                f"Query and candidate sequences must have the same shape, got "
                f"{tuple(query_sequence.shape)} and {tuple(candidate_sequence.shape)}."
            )

        seq_len = query_sequence.shape[0]
        images = torch.cat([query_sequence, candidate_sequence], dim=0).unsqueeze(0).to(self.device)

        if query_points is None:
            query_points = make_grid_query_points(self.image_size, grid_rows=grid_rows, grid_cols=grid_cols)
        query_points = query_points.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                predictions = self.model(images, query_points=query_points)

        source_points = query_points[0]
        target_frame = seq_len + candidate_frame_offset
        target_points = predictions["track"][0, target_frame]
        target_conf = predictions.get("conf")
        target_vis = predictions.get("vis")

        if target_conf is not None:
            target_conf = target_conf[0, target_frame]
        if target_vis is not None:
            target_vis = target_vis[0, target_frame]

        score = score_tracks_with_fundamental_ransac(
            source_points,
            target_points,
            confidence=target_conf,
            visibility=target_vis,
            ransac_reproj_threshold=ransac_reproj_threshold,
        )

        return GeometryVerificationResult(
            score=score,
            raw_predictions={key: value.detach().cpu() for key, value in predictions.items() if torch.is_tensor(value)},
            load_missing_keys=self.missing_keys,
            load_unexpected_keys=self.unexpected_keys,
        )

    def verify_cross_token_3d_pair(
        self,
        query_sequence,
        candidate_sequence,
        grid_rows=12,
        grid_cols=16,
        ransac_threshold_m=0.5,
        ransac_iterations=256,
        seed=0,
    ):
        """Run frozen VGGT once and score 3D center-token/candidate-track matches."""
        if query_sequence.ndim != 4 or candidate_sequence.ndim != 4:
            raise ValueError("Expected query and candidate sequences with shape [S, 3, H, W].")
        if query_sequence.shape != candidate_sequence.shape:
            raise ValueError(
                f"Query and candidate sequences must have the same shape, got "
                f"{tuple(query_sequence.shape)} and {tuple(candidate_sequence.shape)}."
            )
        sequence_length = query_sequence.shape[0]
        query_points = make_grid_query_points(
            self.image_size, grid_rows=grid_rows, grid_cols=grid_cols
        ).unsqueeze(0).to(self.device)
        images = torch.cat([query_sequence, candidate_sequence], dim=0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                predictions = self.model(images, query_points=query_points)
        score = score_cross_token_3d(
            predictions,
            sequence_length=sequence_length,
            image_size=self.image_size,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            ransac_threshold_m=ransac_threshold_m,
            ransac_iterations=ransac_iterations,
            seed=seed,
        )
        return CrossToken3DVerificationResult(
            score=score,
            load_missing_keys=self.missing_keys,
            load_unexpected_keys=self.unexpected_keys,
        )

    def _load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"VGGT geometry checkpoint not found: {checkpoint_path}. "
                "Pass the official VGGT checkpoint path with --vggt_ckpt."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint

        normalized = {}
        for key, value in state_dict.items():
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            normalized[key] = value

        missing, unexpected = self.model.load_state_dict(normalized, strict=False)
        return list(missing), list(unexpected)
