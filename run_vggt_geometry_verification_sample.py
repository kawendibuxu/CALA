import argparse

from geometry_verification.image_io import collect_image_paths, load_rgb_sequence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VGGT geometry verification on one query/candidate image-sequence pair."
    )
    parser.add_argument("--vggt_ckpt", required=True, help="Path to the official VGGT geometry checkpoint.")
    parser.add_argument("--query_dir", required=True, help="Directory containing query RGB images.")
    parser.add_argument("--candidate_dir", required=True, help="Directory containing candidate RGB images.")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=(392, 518), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--grid_rows", type=int, default=12)
    parser.add_argument("--grid_cols", type=int, default=16)
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    from geometry_verification import VGGTGeometryVerifier

    query_paths = collect_image_paths(args.query_dir, args.seq_len)
    candidate_paths = collect_image_paths(args.candidate_dir, args.seq_len)

    query = load_rgb_sequence(query_paths, args.image_size)
    candidate = load_rgb_sequence(candidate_paths, args.image_size)

    verifier = VGGTGeometryVerifier(
        checkpoint_path=args.vggt_ckpt,
        device=args.device,
        image_size=tuple(args.image_size),
    )
    result = verifier.verify_pair(
        query,
        candidate,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
    )

    print("Query images:")
    for path in query_paths:
        print(f"  {path}")
    print("Candidate images:")
    for path in candidate_paths:
        print(f"  {path}")
    print("Geometry score:")
    print(f"  accepted: {result.score.accepted}")
    print(f"  matches: {result.score.num_matches}")
    print(f"  inliers: {result.score.num_inliers}")
    print(f"  inlier_ratio: {result.score.inlier_ratio:.4f}")
    print(f"  mean_sampson_error: {result.score.mean_sampson_error:.4f}")
    print(f"  mean_track_confidence: {result.score.mean_track_confidence:.4f}")
    print(f"  mean_visibility: {result.score.mean_visibility:.4f}")
    if result.load_missing_keys:
        print(f"Warning: missing checkpoint keys: {len(result.load_missing_keys)}")
    if result.load_unexpected_keys:
        print(f"Warning: unexpected checkpoint keys: {len(result.load_unexpected_keys)}")


if __name__ == "__main__":
    main()
