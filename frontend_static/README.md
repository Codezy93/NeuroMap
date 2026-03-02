# NeuroMap Static Demo Frontend

This folder contains a **static demo-only frontend** for codebase analysis insights.
It now includes:

- static slice explorer (mean probability, prediction, entropy, variance)
- composite overlay controls (alpha + threshold)
- static Plotly 3D scatter graph presets
- architecture/risk/insight dashboard sections

## 3D Logic Note

Static 3D scatter is aligned with `backend/neuromap/api.py`:

- tissue mask: `image > image_threshold`
- red points: `tissue & prediction`
- yellow points: `tissue & ~prediction`

If raw MRI is unavailable in this repository, static generation uses `mean_prob` as the image source proxy and applies a fallback threshold to keep a brain-shaped cloud.

Regenerate scatter presets with:

```bash
python frontend_static/scripts/regenerate_scatter.py
```

## Demo Notice

- This is not connected to training, inference, or live API data.
- This is not a clinical interface.
- It is intended only for architecture and engineering review demos.

## Run

Open `index.html` directly in a browser, or serve this folder with any static server.

For best compatibility with local asset loading, run from repository root:

```bash
python -m http.server 8080
```

Then open `http://localhost:8080/frontend_static/`.
