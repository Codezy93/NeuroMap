from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from tqdm import tqdm

from .data import create_train_val_loaders
from .model import build_unet


def _one_hot_from_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    labels = labels.long()
    one_hot = F.one_hot(labels, num_classes=num_classes)
    return one_hot.permute(0, 4, 1, 2, 3).float()


def _build_grad_scaler(enabled: bool):
    """Create GradScaler with compatibility across torch AMP API versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def add_args(parser: argparse.ArgumentParser) -> None:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_workers = 0 if os.name == "nt" else 4
    default_cache_rate = 0.0 if os.name == "nt" else 0.1
    default_val_cache_rate = 0.0 if os.name == "nt" else 0.25
    parser.add_argument("--data-list", required=True, help="Path to MONAI/Decathlon datalist JSON.")
    parser.add_argument("--output-dir", default="artifacts/train", help="Directory for checkpoints.")
    parser.add_argument("--train-key", default="training", help="Training key in datalist JSON.")
    parser.add_argument("--val-key", default="validation", help="Validation key in datalist JSON.")
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--cache-rate", type=float, default=default_cache_rate)
    parser.add_argument("--val-cache-rate", type=float, default=default_val_cache_rate)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--samples-per-volume", type=int, default=4)
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="Enable mixed precision training.",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="neuromap")
    parser.add_argument("--wandb-run-name", default=None)


def run(args: argparse.Namespace) -> None:
    if args.out_channels < 2:
        raise ValueError("out_channels must be >= 2 for softmax-based tumor segmentation.")

    data_list_path = Path(args.data_list).expanduser()
    if not data_list_path.is_absolute():
        data_list_path = (Path.cwd() / data_list_path).resolve()
    if not data_list_path.exists():
        raise FileNotFoundError(
            f"--data-list path does not exist: '{data_list_path}'. "
            "If you pass paths from JSON config (for example VS Code launch.json), "
            "use forward slashes ('data/brats_datalist.json') or escaped backslashes "
            "('data\\\\brats_datalist.json')."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(seed=args.seed)

    if os.name == "nt" and args.workers > 0:
        print(
            "[warning] Windows + DataLoader workers>0 may cause crashes on some setups. "
            "If you see segmentation faults, retry with --workers 0."
        )

    train_loader, val_loader = create_train_val_loaders(
        data_list_path=data_list_path,
        train_key=args.train_key,
        val_key=args.val_key,
        patch_size=args.patch_size,
        samples_per_volume=args.samples_per_volume,
        spacing=args.spacing,
        batch_size=args.batch_size,
        workers=args.workers,
        cache_rate=args.cache_rate,
        val_cache_rate=args.val_cache_rate,
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(device.index or 0)
            print(f"[device] Using GPU via CUDA: {gpu_name}")
        else:
            print("[device] CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
            args.device = "cpu"
            args.amp = False
    else:
        print("[device] Using CPU")

    model = build_unet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        dropout=args.dropout,
    ).to(device)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = _build_grad_scaler(amp_enabled)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except ImportError:
            print("wandb is not installed. Continuing without experiment logging.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )

    history: list[dict[str, float | int | None]] = []
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        step_count = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()

            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())
            step_count += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / max(step_count, 1)
        val_dice = None

        if epoch % args.val_interval == 0:
            model.eval()
            dice_metric.reset()
            with torch.inference_mode():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    logits = sliding_window_inference(
                        images,
                        roi_size=tuple(args.roi_size),
                        sw_batch_size=args.sw_batch_size,
                        predictor=model,
                        overlap=args.overlap,
                    )
                    preds = torch.argmax(logits, dim=1)
                    pred_one_hot = _one_hot_from_labels(preds, args.out_channels)
                    label_one_hot = _one_hot_from_labels(labels, args.out_channels)
                    dice_metric(y_pred=pred_one_hot, y=label_one_hot)

            val_dice = float(dice_metric.aggregate().item())
            dice_metric.reset()

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "best_dice": best_dice,
                        "args": vars(args),
                    },
                    output_dir / "best_model.pt",
                )

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
                "args": vars(args),
            },
            output_dir / "last_model.pt",
        )

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_dice": val_dice,
            "best_dice": best_dice,
        }
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if wandb_run is not None:
            log_payload: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "best_dice": best_dice,
            }
            if val_dice is not None:
                log_payload["val_dice"] = val_dice
            wandb_run.log(log_payload)

        metric_text = f"{val_dice:.4f}" if val_dice is not None else "n/a"
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | val_dice={metric_text} | best_dice={best_dice:.4f}"
        )

    if wandb_run is not None:
        wandb_run.finish()

    print(f"Training complete. Best checkpoint: {output_dir / 'best_model.pt'}")
