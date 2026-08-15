import argparse

from kitti_loop.dataset import KITTILoopSequenceDataset, loop_label, pose_distance


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect KITTI odometry sequence windows.")
    parser.add_argument("--root", default="/data1/jiaming/data-0102/dataset")
    parser.add_argument("--sequence_id", default="00")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image_size", nargs=2, type=int, default=(392, 518), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--query_index", type=int, default=200)
    parser.add_argument("--candidate_index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = KITTILoopSequenceDataset(
        root=args.root,
        sequence_id=args.sequence_id,
        seq_len=args.seq_len,
        stride=args.stride,
        image_size=tuple(args.image_size),
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Sequence: {dataset.sequence_id}")
    print(f"Image dir: {dataset.image_dir}")
    print(f"Pose file: {dataset.pose_path}")
    print(f"Calib keys: {sorted(dataset.calibration.keys())}")

    first = dataset[0]
    print(f"First center_idx: {first.center_idx}")
    print(f"First frame_indices: {first.frame_indices}")
    print(f"First images shape: {tuple(first.images.shape)}")
    print(f"First translation: {first.translation.tolist()}")
    print("First image paths:")
    for path in first.image_paths:
        print(f"  {path}")

    query_index = min(args.query_index, len(dataset) - 1)
    candidate_index = min(args.candidate_index, len(dataset) - 1)
    query = dataset[query_index]
    candidate = dataset[candidate_index]
    distance_m = pose_distance(query.translation, candidate.translation)
    temporal_gap = abs(query.center_idx - candidate.center_idx)
    label = loop_label(distance_m, temporal_gap)
    print("Pair example:")
    print(f"  query_center_idx: {query.center_idx}")
    print(f"  candidate_center_idx: {candidate.center_idx}")
    print(f"  temporal_gap: {temporal_gap}")
    print(f"  distance_m: {distance_m:.3f}")
    print(f"  label: {label}")


if __name__ == "__main__":
    main()
