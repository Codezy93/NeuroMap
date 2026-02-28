from __future__ import annotations

from pathlib import Path
from typing import Sequence

from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.transforms import (
    Compose,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
)


def _binarize_label(label):
    binary = label > 0
    if hasattr(binary, "astype"):
        return binary.astype(label.dtype)
    return binary.to(dtype=label.dtype)


def build_train_transforms(
    patch_size: Sequence[int],
    samples_per_volume: int,
    spacing: Sequence[float],
) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(spacing),
                mode=("bilinear", "nearest"),
            ),
            Lambdad(keys="label", func=_binarize_label),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(patch_size),
                pos=1,
                neg=1,
                num_samples=samples_per_volume,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_val_transforms(spacing: Sequence[float]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(spacing),
                mode=("bilinear", "nearest"),
            ),
            Lambdad(keys="label", func=_binarize_label),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_infer_transforms(spacing: Sequence[float]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first=True),
            Orientationd(keys=["image"], axcodes="RAS", labels=None),
            Spacingd(keys=["image"], pixdim=tuple(spacing), mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )


def create_train_val_loaders(
    data_list_path: str | Path,
    train_key: str,
    val_key: str,
    patch_size: Sequence[int],
    samples_per_volume: int,
    spacing: Sequence[float],
    batch_size: int,
    workers: int,
    cache_rate: float,
    val_cache_rate: float,
) -> tuple[DataLoader, DataLoader]:
    train_files = load_decathlon_datalist(str(data_list_path), True, train_key)
    val_files = load_decathlon_datalist(str(data_list_path), True, val_key)
    if not train_files:
        raise ValueError(f"No training samples found in key '{train_key}'.")
    if not val_files:
        raise ValueError(f"No validation samples found in key '{val_key}'.")

    train_ds = CacheDataset(
        data=train_files,
        transform=build_train_transforms(patch_size, samples_per_volume, spacing),
        cache_rate=cache_rate,
        num_workers=workers,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=build_val_transforms(spacing),
        cache_rate=val_cache_rate,
        num_workers=workers,
    )

    pin_memory = False
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
