from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from typing import Literal

import nibabel as nib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field


@dataclass
class CaseData:
    image: np.ndarray
    prediction: np.ndarray
    entropy: np.ndarray
    image_window: tuple[float, float]
    entropy_max: float


CASES: dict[str, CaseData] = {}


class RegisterCaseRequest(BaseModel):
    case_id: str = Field(min_length=1)
    image_path: str
    prediction_path: str
    entropy_path: str
    image_channel: int = 0
    image_channel_axis: Literal["first", "last"] = "last"


app = FastAPI(title="NeuroMap Slice API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_volume(path: str) -> np.ndarray:
    try:
        return nib.load(path).get_fdata(dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load '{path}': {exc}") from exc


def _as_3d(
    volume: np.ndarray,
    channel: int,
    channel_axis: Literal["first", "last"],
    name: str,
) -> np.ndarray:
    if volume.ndim == 3:
        return volume

    if volume.ndim != 4:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be 3D or 4D, got shape={tuple(volume.shape)}.",
        )

    if channel_axis == "last":
        if channel < 0 or channel >= volume.shape[-1]:
            raise HTTPException(status_code=400, detail=f"Invalid channel index for {name}.")
        return volume[..., channel]

    if channel < 0 or channel >= volume.shape[0]:
        raise HTTPException(status_code=400, detail=f"Invalid channel index for {name}.")
    return volume[channel]


def _compute_window(volume: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(volume, [1.0, 99.0])
    if float(high - low) < 1e-6:
        high = low + 1.0
    return float(low), float(high)


def _slice_to_png(slice_u8: np.ndarray) -> bytes:
    image = Image.fromarray(slice_u8.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _render_image_slice(slice_2d: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    low, high = window
    scaled = (slice_2d - low) / (high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _render_prediction_slice(slice_2d: np.ndarray) -> np.ndarray:
    if float(slice_2d.max()) <= 1.0:
        mask = slice_2d > 0.5
    else:
        mask = slice_2d > 0
    return (mask.astype(np.uint8) * 255).astype(np.uint8)


def _render_entropy_slice(slice_2d: np.ndarray, entropy_max: float) -> np.ndarray:
    if entropy_max <= 1e-8:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    scaled = slice_2d / entropy_max
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _prediction_mask(prediction: np.ndarray) -> np.ndarray:
    max_val = float(np.max(prediction))
    if max_val <= 0.0:
        return np.zeros_like(prediction, dtype=bool)
    if max_val <= 1.0:
        return prediction > 0.5
    return prediction > 0


def _sample_coords(coords: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if coords.shape[0] <= limit:
        return coords
    indices = rng.choice(coords.shape[0], size=limit, replace=False)
    return coords[indices]


def _coords_to_xyz(coords: np.ndarray, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coords.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty
    sx, sy, sz = shape
    denom_x = float(max(sx - 1, 1))
    denom_y = float(max(sy - 1, 1))
    denom_z = float(max(sz - 1, 1))
    x = (coords[:, 0].astype(np.float32) / denom_x - 0.5) * 2.0
    y = (0.5 - coords[:, 1].astype(np.float32) / denom_y) * 2.0
    z = (coords[:, 2].astype(np.float32) / denom_z - 0.5) * 2.0
    return x, y, z


def _case_or_404(case_id: str) -> CaseData:
    case = CASES.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Unknown case_id '{case_id}'.")
    return case


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/cases/register")
@app.post("/cases/create")
def register_case(req: RegisterCaseRequest) -> dict[str, int | str | list[int]]:
    image = _as_3d(
        _load_volume(req.image_path),
        req.image_channel,
        req.image_channel_axis,
        name="image",
    )
    prediction = _as_3d(
        _load_volume(req.prediction_path),
        channel=0,
        channel_axis="last",
        name="prediction",
    )
    entropy = _as_3d(
        _load_volume(req.entropy_path),
        channel=0,
        channel_axis="last",
        name="entropy",
    )

    if image.shape != prediction.shape or image.shape != entropy.shape:
        raise HTTPException(
            status_code=400,
            detail=(
                "Shape mismatch. "
                f"image={tuple(image.shape)}, prediction={tuple(prediction.shape)}, entropy={tuple(entropy.shape)}"
            ),
        )

    window = _compute_window(image)
    entropy_max = float(np.max(entropy))
    CASES[req.case_id] = CaseData(
        image=image,
        prediction=prediction,
        entropy=entropy,
        image_window=window,
        entropy_max=entropy_max,
    )

    return {
        "case_id": req.case_id,
        "shape": [int(v) for v in image.shape],
        "z_slices": int(image.shape[-1]),
    }


@app.get("/cases/{case_id}/meta")
def case_meta(case_id: str) -> dict[str, int | str | list[int]]:
    case = _case_or_404(case_id)
    return {
        "case_id": case_id,
        "shape": [int(v) for v in case.image.shape],
        "z_slices": int(case.image.shape[-1]),
    }


@app.get("/cases/{case_id}/slice/{z}")
def case_slice(
    case_id: str,
    z: int,
    layer: Literal["image", "prediction", "entropy"] = Query("image"),
) -> StreamingResponse:
    case = _case_or_404(case_id)
    max_z = int(case.image.shape[-1]) - 1
    if z < 0 or z > max_z:
        raise HTTPException(status_code=400, detail=f"z must be in [0, {max_z}].")

    if layer == "image":
        slice_u8 = _render_image_slice(case.image[:, :, z], case.image_window)
    elif layer == "prediction":
        slice_u8 = _render_prediction_slice(case.prediction[:, :, z])
    else:
        slice_u8 = _render_entropy_slice(case.entropy[:, :, z], case.entropy_max)

    png_bytes = _slice_to_png(np.rot90(slice_u8))
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/cases/{case_id}/scatter3d")
def case_scatter3d(
    case_id: str,
    image_threshold: float = Query(0.0, ge=0.0),
    slice_stride: int = Query(4, ge=1, le=10),
    pixel_stride: int = Query(4, ge=1, le=10),
    max_points: int = Query(20000, ge=1000, le=500000),
    seed: int = Query(42, ge=0, le=2_147_483_647),
) -> dict[str, int | float | str | dict[str, int] | dict[str, list[float]]]:
    case = _case_or_404(case_id)
    image = case.image[::pixel_stride, ::pixel_stride, ::slice_stride]
    prediction = case.prediction[::pixel_stride, ::pixel_stride, ::slice_stride]

    tissue_mask = image > image_threshold
    pred_mask = _prediction_mask(prediction)
    red_mask = tissue_mask & pred_mask
    yellow_mask = tissue_mask & ~pred_mask

    red_coords = np.argwhere(red_mask)
    yellow_coords = np.argwhere(yellow_mask)

    rng = np.random.default_rng(seed)
    if red_coords.shape[0] >= max_points:
        red_coords = _sample_coords(red_coords, max_points, rng)
        yellow_coords = np.zeros((0, 3), dtype=np.int32)
    else:
        remaining = max_points - red_coords.shape[0]
        yellow_coords = _sample_coords(yellow_coords, remaining, rng)

    x_red, y_red, z_red = _coords_to_xyz(red_coords, image.shape)
    x_yel, y_yel, z_yel = _coords_to_xyz(yellow_coords, image.shape)
    point_size = float(np.clip(0.7 + 0.22 * pixel_stride + 0.18 * slice_stride, 0.8, 2.8))

    red_count = int(red_coords.shape[0])
    yellow_count = int(yellow_coords.shape[0])
    return {
        "case_id": case_id,
        "point_size": point_size,
        "counts": {
            "red": red_count,
            "yellow": yellow_count,
            "total": int(red_count + yellow_count),
        },
        "red": {
            "x": x_red.tolist(),
            "y": y_red.tolist(),
            "z": z_red.tolist(),
        },
        "yellow": {
            "x": x_yel.tolist(),
            "y": y_yel.tolist(),
            "z": z_yel.tolist(),
        },
    }


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")


def run(args: argparse.Namespace) -> None:
    if args.reload:
        uvicorn.run("neuromap.api:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)
