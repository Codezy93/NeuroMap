from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .data import build_infer_transforms
from .model import build_unet
from .uq import mc_dropout_predict


def _extract_affine(sample: dict) -> np.ndarray:
    meta = sample.get("image_meta_dict")
    if isinstance(meta, dict) and "affine" in meta:
        affine = np.asarray(meta["affine"])
        if affine.ndim == 3:
            affine = affine[0]
        return affine.astype(np.float32)

    image = sample.get("image")
    if hasattr(image, "affine"):
        affine = np.asarray(image.affine)
        if affine.ndim == 3:
            affine = affine[0]
        return affine.astype(np.float32)

    return np.eye(4, dtype=np.float32)


def _save_nifti(volume: np.ndarray, affine: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume, affine), str(path))


def add_args(parser: argparse.ArgumentParser) -> None:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--images", nargs="+", required=True, help="Input MRI modalities for one case.")
    parser.add_argument("--output-dir", default="artifacts/predict")
    parser.add_argument("--output-prefix", default="case")
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tumor-channel", type=int, default=1)
    parser.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--out-channels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        ckpt_args = checkpoint.get("args", {})
    else:
        state_dict = checkpoint
        ckpt_args = {}

    transforms = build_infer_transforms(args.spacing)
    sample = transforms({"image": args.images})
    image = sample["image"].unsqueeze(0).to(device)
    affine = _extract_affine(sample)

    inferred_in_channels = image.shape[1]
    in_channels = args.in_channels or ckpt_args.get("in_channels", inferred_in_channels)
    out_channels = args.out_channels or ckpt_args.get("out_channels", 2)
    dropout = args.dropout if args.dropout is not None else ckpt_args.get("dropout", 0.2)

    if args.tumor_channel >= out_channels:
        raise ValueError("tumor_channel must be smaller than out_channels.")

    model = build_unet(in_channels=in_channels, out_channels=out_channels, dropout=dropout).to(device)
    model.load_state_dict(state_dict)

    mean_prob, variance, entropy = mc_dropout_predict(
        model=model,
        image=image,
        mc_samples=args.mc_samples,
        roi_size=args.roi_size,
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        tumor_channel=args.tumor_channel,
    )

    mean_prob_np = mean_prob.squeeze(0).cpu().numpy().astype(np.float32)
    variance_np = variance.squeeze(0).cpu().numpy().astype(np.float32)
    entropy_np = entropy.squeeze(0).cpu().numpy().astype(np.float32)
    prediction_np = (mean_prob_np >= args.threshold).astype(np.uint8)

    output_dir = Path(args.output_dir)
    prefix = args.output_prefix
    pred_path = output_dir / f"{prefix}_prediction.nii.gz"
    prob_path = output_dir / f"{prefix}_mean_prob.nii.gz"
    var_path = output_dir / f"{prefix}_variance.nii.gz"
    ent_path = output_dir / f"{prefix}_entropy.nii.gz"

    _save_nifti(prediction_np, affine, pred_path)
    _save_nifti(mean_prob_np, affine, prob_path)
    _save_nifti(variance_np, affine, var_path)
    _save_nifti(entropy_np, affine, ent_path)

    print(f"Saved prediction: {pred_path}")
    print(f"Saved mean probability: {prob_path}")
    print(f"Saved variance map: {var_path}")
    print(f"Saved entropy map: {ent_path}")
