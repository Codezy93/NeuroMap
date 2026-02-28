# NeuroMap

NeuroMap is a 3D volumetric brain tumor segmentation project with uncertainty quantification (UQ).  
Instead of returning only a hard mask, it also produces uncertainty volumes (variance + entropy) from Monte Carlo dropout.

## What It Includes

- MONAI + PyTorch training pipeline for 3D MRI segmentation
- Patch-based training with `CacheDataset` + `RandCropByPosNegLabeld`
- 3D U-Net with dropout kept active for MC-dropout inference
- Uncertainty outputs:
  - `mean_prob`: average tumor probability across stochastic passes
  - `variance`: voxel-wise predictive variance
  - `entropy`: voxel-wise predictive entropy
- FastAPI backend for slice-serving (`image`, `prediction`, `entropy`)
- Next.js frontend for interactive Z-slice review and overlays

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Frontend setup:

```bash
cd frontend
npm.cmd install
```

## Data Format (Decathlon/MONAI JSON)

Use a datalist JSON with `training` and `validation` keys.

```json
{
  "training": [
    {
      "image": [
        "imagesTr/BraTS_001_t1.nii.gz",
        "imagesTr/BraTS_001_t1ce.nii.gz",
        "imagesTr/BraTS_001_t2.nii.gz",
        "imagesTr/BraTS_001_flair.nii.gz"
      ],
      "label": "labelsTr/BraTS_001_seg.nii.gz"
    }
  ],
  "validation": [
    {
      "image": [
        "imagesVal/BraTS_101_t1.nii.gz",
        "imagesVal/BraTS_101_t1ce.nii.gz",
        "imagesVal/BraTS_101_t2.nii.gz",
        "imagesVal/BraTS_101_flair.nii.gz"
      ],
      "label": "labelsVal/BraTS_101_seg.nii.gz"
    }
  ]
}
```

## Train

```bash
python -m neuromap train --data-list data/brats_datalist.json --output-dir artifacts/run01 --in-channels 4 --out-channels 2 --epochs 200 --batch-size 2 --samples-per-volume 4 --patch-size 96 96 96 --use-wandb
```

Windows/Git-Bash stability profile (recommended if you hit crashes):

```bash
python -m neuromap train --data-list src/data/brats_datalist.json --output-dir artifacts/run01 --in-channels 4 --out-channels 2 --epochs 200 --batch-size 1 --samples-per-volume 2 --patch-size 96 96 96 --workers 0 --cache-rate 0.0 --val-cache-rate 0.0 --no-amp
```

## MC-Dropout Inference

```bash
python -m neuromap predict ^
  --checkpoint artifacts\run01\best_model.pt ^
  --images data\case001_t1.nii.gz data\case001_t1ce.nii.gz data\case001_t2.nii.gz data\case001_flair.nii.gz ^
  --output-dir artifacts\predictions ^
  --output-prefix case001 ^
  --mc-samples 20
```

Outputs:

- `case001_prediction.nii.gz`
- `case001_mean_prob.nii.gz`
- `case001_variance.nii.gz`
- `case001_entropy.nii.gz`

## Start API

```bash
python -m neuromap serve --host 0.0.0.0 --port 8000
```

Core endpoints:

- `GET /health`
- `POST /cases/register`
- `GET /cases/{case_id}/meta`
- `GET /cases/{case_id}/slice/{z}?layer=image|prediction|entropy`
- `GET /cases/{case_id}/scatter3d`

## Start Next.js Frontend

```bash
cd frontend
npm.cmd run dev
```

Frontend capabilities:

- API health check button
- Register a new case with channel options
- Load an existing registered case by `case_id`
- Composite view or side-by-side MRI / prediction / entropy panels
- Slice navigation (`Prev`, slider, `Next`)
- Alpha and threshold controls for prediction and uncertainty overlays
- Slice-level metrics (tumor pixel %, hotspot pixel %, mean uncertainty)
- Plotly 3D scatter reconstruction from volume:
  - black/background voxels are skipped
  - non-black tissue voxels are yellow points
  - prediction voxels are red points
- Same-origin Next.js API bridge to FastAPI (helps avoid browser extension blocking on direct `localhost:8000` calls)

In the sidebar, register a case by pointing to:

- raw MRI volume (`image_path`)
- predicted mask volume (`prediction_path`)
- entropy volume (`entropy_path`)

Then use the slice navigation and overlay controls in the main pane.

## Notes

- Training and inference are 3D and memory-intensive. Tune `--patch-size`, `--roi-size`, and `--sw-batch-size` for your GPU.
- MC-dropout is enabled by forcing dropout layers to `train()` during inference while keeping the rest of the model in eval mode.
- The current setup uses binary tumor vs background (`out_channels=2`).
