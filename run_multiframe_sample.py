import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from vggt.models.vggtpr_lora import VGGTPR_LoRA


DEFAULT_IMAGE_DIR = "/data1/jiaming/test/stockholm/query/images"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one UniPR-3D multi-frame forward pass without dataset metrics."
    )
    parser.add_argument(
        "--ckpt",
        default="model/multi_model.ckpt",
        help="Path to the multi-frame checkpoint.",
    )
    parser.add_argument(
        "--image_dir",
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing sample images. Ignored when --images is provided.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Explicit image paths to use as one sequence.",
    )
    parser.add_argument("--seq_len", type=int, default=5, help="Number of frames in the sequence.")
    parser.add_argument(
        "--image_size",
        nargs=2,
        type=int,
        default=(392, 518),
        metavar=("HEIGHT", "WIDTH"),
        help="Resize images to this H W before inference.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for inference.",
    )
    return parser.parse_args()


def collect_images(image_dir, explicit_images, seq_len):
    if explicit_images:
        paths = [Path(p) for p in explicit_images]
    else:
        root = Path(image_dir)
        exts = {".jpg", ".jpeg", ".png"}
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)

    if len(paths) < seq_len:
        raise ValueError(f"Need at least {seq_len} images, found {len(paths)}.")

    selected = paths[:seq_len]
    missing = [str(p) for p in selected if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image files: {missing}")

    return selected


def load_sequence(paths, image_size):
    transform = T.Compose(
        [
            T.Resize(tuple(image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ]
    )
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(transform(image.convert("RGB")))
    return torch.stack(frames, dim=0).unsqueeze(0)


def build_model_from_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    state_dict = checkpoint.get("state_dict", checkpoint)

    model = VGGTPR_LoRA(
        lora_rank=hparams.get("lora_rank", 8),
        lora_alpha=hparams.get("lora_alpha", 16),
        lora_dropout=hparams.get("lora_dropout", 0.1),
        lora_frame_attn=hparams.get("lora_frame_attn", True),
        lora_global_attn=hparams.get("lora_global_attn", True),
        lora_patch_embed=hparams.get("lora_patch_embed", False),
        with_camera_pose=hparams.get("with_camera_pose", False),
        camera_pose_type=hparams.get("camera_pose_type", "yaw"),
        with_dinov2_features=hparams.get("with_dinov2_features", True),
        with_geo_features=True,
    )

    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key.removeprefix("model.")] = value

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing keys, first 5: {missing[:5]}")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys, first 5: {unexpected[:5]}")
    return model


def main():
    args = parse_args()
    image_paths = collect_images(args.image_dir, args.images, args.seq_len)
    images = load_sequence(image_paths, args.image_size)

    print("Selected images:")
    for path in image_paths:
        print(f"  {path}")
    print(f"Input tensor shape: {tuple(images.shape)}")

    model = build_model_from_checkpoint(args.ckpt)
    model.eval().to(args.device)
    images = images.to(args.device)

    with torch.inference_mode():
        with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=args.device == "cuda"):
            output = model(images)
            descriptor = output["salad_pred"]

    descriptor_cpu = descriptor.detach().float().cpu()
    print(f"Descriptor shape: {tuple(descriptor_cpu.shape)}")
    print(f"Descriptor L2 norm: {descriptor_cpu.norm(dim=1).tolist()}")
    print(f"Descriptor first 8 values: {descriptor_cpu[0, :8].tolist()}")


if __name__ == "__main__":
    main()
