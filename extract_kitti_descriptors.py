import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kitti_loop.dataset import KITTILoopSequenceDataset
from run_multiframe_sample import build_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Extract UniPR-3D multi-frame descriptors on KITTI.")
    parser.add_argument("--root", default="/data1/jiaming/data-0102/dataset")
    parser.add_argument("--sequence_id", default="00")
    parser.add_argument("--ckpt", default="model/multi_model.ckpt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image_size", nargs=2, type=int, default=(392, 518), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    return parser.parse_args()


def collate_kitti_samples(samples):
    return {
        "images": torch.stack([sample.images for sample in samples], dim=0),
        "center_indices": torch.tensor([sample.center_idx for sample in samples], dtype=torch.long),
        "frame_indices": torch.tensor([sample.frame_indices for sample in samples], dtype=torch.long),
        "poses": torch.stack([sample.pose for sample in samples], dim=0),
        "translations": torch.stack([sample.translation for sample in samples], dim=0),
        "image_paths": [sample.image_paths for sample in samples],
        "sequence_ids": [sample.sequence_id for sample in samples],
    }


def default_output_path(sequence_id, seq_len, stride):
    return Path("outputs") / f"kitti_{sequence_id}_seq{seq_len}_stride{stride}_unipr_descriptors.pt"


def main():
    args = parse_args()
    output_path = Path(args.output) if args.output else default_output_path(args.sequence_id, args.seq_len, args.stride)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = KITTILoopSequenceDataset(
        root=args.root,
        sequence_id=args.sequence_id,
        seq_len=args.seq_len,
        stride=args.stride,
        image_size=tuple(args.image_size),
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        collate_fn=collate_kitti_samples,
    )

    model = build_model_from_checkpoint(args.ckpt)
    model.eval().to(args.device)

    descriptor_chunks = []
    center_indices = []
    frame_indices = []
    poses = []
    translations = []
    image_paths = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Extracting KITTI {dataset.sequence_id} descriptors"):
            images = batch["images"].to(args.device, non_blocking=True)
            with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=args.device == "cuda"):
                descriptors = model(images)["salad_pred"]

            descriptor_chunks.append(descriptors.detach().float().cpu())
            center_indices.append(batch["center_indices"])
            frame_indices.append(batch["frame_indices"])
            poses.append(batch["poses"])
            translations.append(batch["translations"])
            image_paths.extend(batch["image_paths"])

    data = {
        "sequence_id": dataset.sequence_id,
        "root": str(Path(args.root)),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "image_size": tuple(args.image_size),
        "center_first": dataset.center_first,
        "descriptors": torch.cat(descriptor_chunks, dim=0),
        "center_indices": torch.cat(center_indices, dim=0),
        "frame_indices": torch.cat(frame_indices, dim=0),
        "poses": torch.cat(poses, dim=0),
        "translations": torch.cat(translations, dim=0),
        "image_paths": image_paths,
    }
    torch.save(data, output_path)
    print(f"Saved descriptors to {output_path}")
    print(f"Descriptors shape: {tuple(data['descriptors'].shape)}")
    print(f"Center index range: {int(data['center_indices'][0])} - {int(data['center_indices'][-1])}")


if __name__ == "__main__":
    main()
