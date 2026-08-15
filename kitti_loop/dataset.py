from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


@dataclass(frozen=True)
class KITTISample:
    sequence_id: str
    center_idx: int
    frame_indices: list[int]
    image_paths: list[str]
    pose: torch.Tensor
    translation: torch.Tensor
    images: torch.Tensor


class KITTILoopSequenceDataset(Dataset):
    def __init__(
        self,
        root,
        sequence_id="00",
        seq_len=5,
        stride=1,
        image_size=(392, 518),
        camera="image_2",
        center_first=True,
        max_samples=None,
    ):
        if seq_len < 1 or seq_len % 2 != 1:
            raise ValueError("seq_len must be an odd positive integer.")

        self.root = Path(root)
        self.sequence_id = f"{int(sequence_id):02d}" if str(sequence_id).isdigit() else str(sequence_id)
        self.seq_len = seq_len
        self.stride = stride
        self.image_size = tuple(image_size)
        self.camera = camera
        self.center_first = center_first

        self.image_dir = self.root / "sequences" / self.sequence_id / camera
        self.pose_path = self.root / "poses" / f"{self.sequence_id}.txt"
        self.calib_path = self.root / "sequences" / self.sequence_id / "calib.txt"

        self.image_paths = self._load_image_paths()
        self.poses = self._load_poses()
        self.calibration = self._load_calibration()

        if len(self.image_paths) != len(self.poses):
            common_len = min(len(self.image_paths), len(self.poses))
            self.image_paths = self.image_paths[:common_len]
            self.poses = self.poses[:common_len]

        half = seq_len // 2
        self.centers = list(range(half, len(self.image_paths) - half, stride))
        if max_samples is not None:
            self.centers = self.centers[:max_samples]

        self.transform = T.Compose(
            [
                T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ]
        )

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, index):
        center = self.centers[index]
        half = self.seq_len // 2
        frame_indices = list(range(center - half, center + half + 1))

        if self.center_first:
            frame_indices = [center] + frame_indices[:half] + frame_indices[half + 1:]

        paths = [self.image_paths[i] for i in frame_indices]
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.transform(image.convert("RGB")))

        pose = torch.from_numpy(self.poses[center]).float()
        translation = pose[:3, 3]

        return KITTISample(
            sequence_id=self.sequence_id,
            center_idx=center,
            frame_indices=frame_indices,
            image_paths=[str(p) for p in paths],
            pose=pose,
            translation=translation,
            images=torch.stack(images, dim=0),
        )

    def distance(self, query_dataset_index, candidate_dataset_index, axes=(0, 2)):
        query_t = self[query_dataset_index].translation
        candidate_t = self[candidate_dataset_index].translation
        return pose_distance(query_t, candidate_t, axes=axes)

    def _load_image_paths(self):
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"KITTI image directory not found: {self.image_dir}")

        paths = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        return paths

    def _load_poses(self):
        if not self.pose_path.is_file():
            raise FileNotFoundError(f"KITTI pose file not found: {self.pose_path}")

        poses = []
        for line in self.pose_path.read_text().splitlines():
            values = np.fromstring(line, sep=" ", dtype=np.float32)
            if values.size != 12:
                continue
            poses.append(values.reshape(3, 4))
        if not poses:
            raise ValueError(f"No valid poses in {self.pose_path}")
        return np.stack(poses, axis=0)

    def _load_calibration(self):
        if not self.calib_path.is_file():
            return {}

        calibration = {}
        for line in self.calib_path.read_text().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            arr = np.fromstring(value, sep=" ", dtype=np.float32)
            calibration[key] = arr
        return calibration


def pose_distance(translation_a, translation_b, axes=(0, 2)):
    diff = translation_a[list(axes)] - translation_b[list(axes)]
    return float(torch.linalg.norm(diff).item())


def loop_label(distance_m, temporal_gap, positive_radius=5.0, negative_radius=25.0, min_temporal_gap=100):
    if temporal_gap < min_temporal_gap:
        return "ignore"
    if distance_m <= positive_radius:
        return "positive"
    if distance_m >= negative_radius:
        return "negative"
    return "ignore"
