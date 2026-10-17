"""Test GCN-only and Mamba-only GraphMamba branch ablations on MSRS."""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import fusion_dataset
from model_gcn import backbone as GCNBackbone
from model_mamba import backbone as MambaBackbone
from test_node import (
    fuse_rgb,
    load_checkpoint,
    resolve_input_dir,
    save_batch,
    validate_pairs,
)


DEFAULT_TEST_ROOT = "/mnt/ssd1/home/yustudent/students/zjw/MSRS/test"
DEFAULT_RESULTS_ROOT = (
    "/mnt/ssd1/home/yustudent/students/zjw/gcn+mamba_results"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test GCN-only or Mamba-only GMFuse on MSRS"
    )
    parser.add_argument(
        "--branch",
        choices=("gcn", "mamba"),
        required=True,
        help="Branch-only model architecture to instantiate",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the matching fusion/fusion_best.pth",
    )
    parser.add_argument("--test-root", default=DEFAULT_TEST_ROOT)
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--ir-dir", default=None)
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root under which gcn_only/ or mamba_only/ is created",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-seg-classes", type=int, default=21)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested ({args.device}), but CUDA is unavailable"
        )
    device = torch.device(args.device)

    visible_dir = resolve_input_dir(
        args.test_root,
        args.vis_dir,
        ("vi", "vis", "visible", "Visible"),
        "visible-image",
    )
    infrared_dir = resolve_input_dir(
        args.test_root,
        args.ir_dir,
        ("ir", "infrared", "Infrared"),
        "infrared-image",
    )
    validate_pairs(visible_dir, infrared_dir)

    experiment_name = "gcn_only" if args.branch == "gcn" else "mamba_only"
    output_dir = Path(
        args.output_dir or Path(args.results_root) / experiment_name
    )
    existing = (
        list(output_dir.glob("*.png")) if output_dir.is_dir() else []
    )
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} PNG files. "
            "Use --overwrite only if these results should be replaced."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_class = GCNBackbone if args.branch == "gcn" else MambaBackbone
    model = model_class(
        num_seg_classes=args.num_seg_classes,
        hidden_dim_gnn=None,
    ).to(device)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    dataset = fusion_dataset(
        ir_path=str(infrared_dir),
        vis_path=str(visible_dir),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    print(f"Branch: {experiment_name}")
    print(f"Device: {device}")
    print(f"Test pairs: {len(dataset)}")
    print(f"Output: {output_dir}")

    elapsed = 0.0
    with torch.inference_mode():
        for visible, infrared, names in tqdm(
            loader, desc=experiment_name, unit="batch"
        ):
            visible = visible.to(device, non_blocking=True)
            infrared = infrared.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            fused = fuse_rgb(model, visible, infrared)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
            save_batch(fused, names, output_dir)

    print("Testing complete")
    print(f"Images: {len(dataset)}")
    print(f"Pure inference time: {elapsed:.3f} s")
    print(f"Average inference time: {elapsed / max(len(dataset), 1):.6f} s/image")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
