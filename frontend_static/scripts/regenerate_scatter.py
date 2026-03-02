from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PREDICT_DIR = ROOT / "backend" / "artifacts" / "predict"
ASSETS_DIR = ROOT / "frontend_static" / "assets"
MANIFEST_PATH = ASSETS_DIR / "manifest.json"

# Used when raw MRI is unavailable and mean_prob is the only "image-like" layer.
# We adapt this per-volume so tissue points remain visible in static 3D renders.
MEAN_PROB_IMAGE_THRESHOLD_QUANTILE = 0.90


def load_volume(path: Path) -> np.ndarray:
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={tuple(data.shape)} for {path.name}")
    return data


def load_optional_image_source() -> tuple[np.ndarray | None, str | None]:
    """
    Prefer raw MRI when available so yellow points map to brain tissue.
    """
    candidates = [
        PREDICT_DIR / "case001_image.nii.gz",
        PREDICT_DIR / "case001_mri.nii.gz",
        PREDICT_DIR / "case001_input.nii.gz",
        ROOT / "backend" / "data" / "Task01_BrainTumour" / "imagesTr" / "BRATS_001.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            return load_volume(path), str(path.relative_to(ROOT)).replace("\\", "/")
    return None, None


def sample_coords(coords: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if coords.shape[0] <= limit:
        return coords
    idx = rng.choice(coords.shape[0], size=limit, replace=False)
    return coords[idx]


def coords_to_xyz(coords: np.ndarray, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coords.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty
    sx, sy, sz = shape
    # Match backend/neuromap/api.py::_coords_to_xyz exactly.
    denom_x = float(max(sx - 1, 1))
    denom_y = float(max(sy - 1, 1))
    denom_z = float(max(sz - 1, 1))
    x = (coords[:, 0].astype(np.float32) / denom_x - 0.5) * 2.0
    y = (0.5 - coords[:, 1].astype(np.float32) / denom_y) * 2.0
    z = (coords[:, 2].astype(np.float32) / denom_z - 0.5) * 2.0
    return x, y, z


def prediction_mask(prediction: np.ndarray) -> np.ndarray:
    max_val = float(np.max(prediction))
    if max_val <= 0.0:
        return np.zeros_like(prediction, dtype=bool)
    if max_val <= 1.0:
        return prediction > 0.5
    return prediction > 0


def choose_mean_prob_threshold(mean_prob: np.ndarray) -> float:
    """
    Static fallback when raw MRI is unavailable.
    In backend API, `image_threshold` defaults to 0.0 on real MRI volumes.
    For mean_prob fallback, use a high quantile so the proxy image remains brain-shaped.
    """
    q = float(np.quantile(mean_prob, MEAN_PROB_IMAGE_THRESHOLD_QUANTILE))
    return float(np.clip(q, 0.18, 0.35))


def build_scatter_payload(
    *,
    name: str,
    image: np.ndarray,
    prediction: np.ndarray,
    slice_stride: int,
    pixel_stride: int,
    max_points: int,
    seed: int,
    image_threshold: float,
    tissue_source_name: str,
) -> dict:
    image_ds = image[::pixel_stride, ::pixel_stride, ::slice_stride]
    prediction_ds = prediction[::pixel_stride, ::pixel_stride, ::slice_stride]

    # Match backend/neuromap/api.py::case_scatter3d logic exactly.
    tissue = image_ds > float(image_threshold)
    pred = prediction_mask(prediction_ds)

    red_mask = tissue & pred
    yellow_mask = tissue & ~pred

    red_coords = np.argwhere(red_mask)
    yellow_coords = np.argwhere(yellow_mask)

    rng = np.random.default_rng(seed)
    if red_coords.shape[0] >= max_points:
        red_coords = sample_coords(red_coords, max_points, rng)
        yellow_coords = np.zeros((0, 3), dtype=np.int32)
    else:
        yellow_coords = sample_coords(yellow_coords, max_points - red_coords.shape[0], rng)

    x_red, y_red, z_red = coords_to_xyz(red_coords, tissue.shape)
    x_yel, y_yel, z_yel = coords_to_xyz(yellow_coords, tissue.shape)
    point_size = float(np.clip(0.7 + 0.22 * pixel_stride + 0.18 * slice_stride, 0.8, 2.8))

    return {
        "preset": name,
        "point_size": point_size,
        "counts": {
            "red": int(red_coords.shape[0]),
            "yellow": int(yellow_coords.shape[0]),
            "total": int(red_coords.shape[0] + yellow_coords.shape[0]),
        },
        "params": {
            "slice_stride": slice_stride,
            "pixel_stride": pixel_stride,
            "max_points": max_points,
            "seed": seed,
            "image_threshold": image_threshold,
            # Backward-compatible key consumed by current static UI status label.
            "white_threshold": image_threshold,
            "tissue_source": tissue_source_name,
            "mask_logic": "yellow=(image>threshold)&~prediction, red=(image>threshold)&prediction",
        },
        "red": {"x": x_red.tolist(), "y": y_red.tolist(), "z": z_red.tolist()},
        "yellow": {"x": x_yel.tolist(), "y": y_yel.tolist(), "z": z_yel.tolist()},
    }


def main() -> None:
    mean_prob = load_volume(PREDICT_DIR / "case001_mean_prob.nii.gz")
    prediction = load_volume(PREDICT_DIR / "case001_prediction.nii.gz")

    # Keep explicit dependency on generated artifacts.
    _ = load_volume(PREDICT_DIR / "case001_entropy.nii.gz")
    _ = load_volume(PREDICT_DIR / "case001_variance.nii.gz")

    image_source, image_source_path = load_optional_image_source()
    if image_source is not None:
        scatter_image_source = image_source
        tissue_source_name = "image"
        image_threshold = 0.0
    else:
        scatter_image_source = mean_prob
        tissue_source_name = "mean_prob"
        image_threshold = choose_mean_prob_threshold(mean_prob)

    presets = {
        "low": {"slice_stride": 5, "pixel_stride": 5, "max_points": 8000},
        "medium": {"slice_stride": 4, "pixel_stride": 4, "max_points": 12000},
        "high": {"slice_stride": 3, "pixel_stride": 3, "max_points": 22000},
    }

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in presets.items():
        payload = build_scatter_payload(
            name=name,
            image=scatter_image_source,
            prediction=prediction,
            slice_stride=cfg["slice_stride"],
            pixel_stride=cfg["pixel_stride"],
            max_points=cfg["max_points"],
            seed=42,
            image_threshold=image_threshold,
            tissue_source_name=tissue_source_name,
        )
        out_path = ASSETS_DIR / f"scatter3d_{name}.json"
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"Wrote {out_path.name}: {payload['counts']}")

    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["scatterLogic"] = {
            "description": "API-aligned: tissue=image>threshold; red=tissue&prediction; yellow=tissue&~prediction",
            "usesRawMRI": bool(image_source is not None),
            "tissueSource": tissue_source_name,
            "tissueSourcePath": image_source_path,
            "imageThreshold": image_threshold,
            "note": "API-aligned scatter logic. If raw MRI is unavailable, mean_prob is used as image proxy.",
        }
        MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")
        print(f"Updated {MANIFEST_PATH.name} scatter metadata.")


if __name__ == "__main__":
    main()
