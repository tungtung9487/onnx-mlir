#!/usr/bin/env python3
"""Generate the int8-QDQ calibration set val_224_txt/ : one .txt per image, a
single line of space-separated float32 = the flattened, preprocessed 1x3x224x224
tensor. Same transform as the ORT eval / runtime image path
(Resize(256) -> CenterCrop(224) -> ToTensor -> ImageNet mean/std). Images are the
first --limit samples of the sorted ImageFolder (class-ordered, same convention
the original val_224_txt used); ~256-500 is enough for MinMax calibration.

Usage:
  python gen_val_224_txt.py --data-root imagenet100_hf/validation \
      --out-dir val_224_txt --limit 500
"""
import argparse
import os

import numpy as np
from torchvision import datasets, transforms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="imagenet100_hf/validation")
    ap.add_argument("--out-dir", default="val_224_txt")
    ap.add_argument("--limit", type=int, default=500, help="0 = all images")
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    ap.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    args = ap.parse_args()

    tf = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ])
    ds = datasets.ImageFolder(args.data_root, transform=tf)
    os.makedirs(args.out_dir, exist_ok=True)
    n = min(args.limit, len(ds)) if args.limit > 0 else len(ds)
    for i in range(n):
        x, _ = ds[i]
        flat = x.numpy().reshape(-1).astype(np.float32)
        with open(os.path.join(args.out_dir, f"img_{i:05d}.txt"), "w") as f:
            f.write(" ".join(f"{v:.9g}" for v in flat))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    print(f"wrote {n} calibration txt to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
