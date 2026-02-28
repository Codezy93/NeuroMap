"use client";
/* eslint-disable @next/next/no-img-element */

import dynamic from "next/dynamic";
import type { Data, Layout } from "plotly.js";
import { useEffect, useMemo, useRef, useState } from "react";

type Layer = "image" | "prediction" | "entropy";
type DisplayMode = "composite" | "sideBySide" | "imageOnly";
type ChannelAxis = "first" | "last";
type Palette = "ember" | "mint" | "ocean";

type CaseMeta = {
  case_id: string;
  shape: number[];
  z_slices: number;
};

type GrayImage = {
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
};

type LayerImages = {
  image: GrayImage;
  prediction: GrayImage;
  entropy: GrayImage;
};

type LayerUrls = {
  image: string;
  prediction: string;
  entropy: string;
};

type SliceMetrics = {
  tumorPct: number;
  hotspotPct: number;
  entropyMean: number;
};

type ScatterSeries = {
  x: number[];
  y: number[];
  z: number[];
};

type Scatter3DPayload = {
  case_id: string;
  point_size: number;
  counts: {
    red: number;
    yellow: number;
    total: number;
  };
  red: ScatterSeries;
  yellow: ScatterSeries;
};

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const DEFAULT_API_URL = "http://127.0.0.1:8000";
const DEFAULT_CASE_ID = "case_001";

function normalizeApiUrl(url: string): string {
  return url.trim().replace(/\/+$/, "");
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

async function readApiError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload?.detail) {
      return payload.detail;
    }
  } catch {
    // no-op
  }
  return response.statusText || "Request failed";
}

function bridgePath(path: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `/api/bridge${normalized}`;
}

function buildCaseLayerPath(caseId: string, layer: Layer, z: number, cacheKey: number): string {
  return `/cases/${encodeURIComponent(caseId)}/slice/${z}?layer=${layer}&t=${cacheKey}`;
}

function buildProxyHeaders(apiUrl: string, initialHeaders?: HeadersInit): Headers {
  const headers = new Headers(initialHeaders);
  const normalizedApiUrl = normalizeApiUrl(apiUrl);
  if (normalizedApiUrl) {
    headers.set("x-neuromap-api-url", normalizedApiUrl);
  }
  return headers;
}

async function proxyFetch(apiUrl: string, path: string, init?: RequestInit): Promise<Response> {
  return fetch(bridgePath(path), {
    cache: "no-store",
    ...init,
    headers: buildProxyHeaders(apiUrl, init?.headers),
  });
}

async function fetchGrayImage(
  apiUrl: string,
  path: string,
  signal?: AbortSignal,
): Promise<{ gray: GrayImage; blob: Blob }> {
  const response = await proxyFetch(apiUrl, path, { signal });
  if (!response.ok) {
    throw new Error(await readApiError(response));
  }
  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);

  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Could not initialize 2D canvas context.");
  }
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();

  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const grayPixels = new Uint8ClampedArray(canvas.width * canvas.height);
  for (let i = 0, j = 0; i < grayPixels.length; i++, j += 4) {
    grayPixels[i] = rgba[j];
  }
  return {
    gray: { width: canvas.width, height: canvas.height, pixels: grayPixels },
    blob,
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpColor(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [
    Math.round(lerp(a[0], b[0], t)),
    Math.round(lerp(a[1], b[1], t)),
    Math.round(lerp(a[2], b[2], t)),
  ];
}

function paletteColor(palette: Palette, t: number): [number, number, number] {
  const c = clamp(t, 0, 1);
  if (palette === "mint") {
    if (c < 0.5) return lerpColor([6, 78, 59], [16, 185, 129], c * 2);
    return lerpColor([16, 185, 129], [110, 231, 183], (c - 0.5) * 2);
  }
  if (palette === "ocean") {
    if (c < 0.5) return lerpColor([8, 47, 73], [2, 132, 199], c * 2);
    return lerpColor([2, 132, 199], [186, 230, 253], (c - 0.5) * 2);
  }
  if (c < 0.5) return lerpColor([120, 53, 15], [249, 115, 22], c * 2);
  return lerpColor([249, 115, 22], [251, 191, 36], (c - 0.5) * 2);
}

function blend(base: [number, number, number], overlay: [number, number, number], alpha: number): [number, number, number] {
  const a = clamp(alpha, 0, 1);
  return [
    Math.round(base[0] * (1 - a) + overlay[0] * a),
    Math.round(base[1] * (1 - a) + overlay[1] * a),
    Math.round(base[2] * (1 - a) + overlay[2] * a),
  ];
}

function buildCompositeDataUrl(
  layers: LayerImages,
  options: {
    showPrediction: boolean;
    showEntropy: boolean;
    predictionAlpha: number;
    entropyAlpha: number;
    predictionThreshold: number;
    entropyThreshold: number;
    predictionPalette: Palette;
    entropyPalette: Palette;
  },
): string {
  const { image, prediction, entropy } = layers;
  const total = image.pixels.length;
  const rgba = new Uint8ClampedArray(total * 4);

  for (let i = 0, j = 0; i < total; i++, j += 4) {
    const base = image.pixels[i] / 255;
    let color: [number, number, number] = [Math.round(base * 255), Math.round(base * 255), Math.round(base * 255)];

    if (options.showPrediction) {
      const v = prediction.pixels[i] / 255;
      if (v >= options.predictionThreshold) {
        color = blend(color, paletteColor(options.predictionPalette, v), options.predictionAlpha * v);
      }
    }
    if (options.showEntropy) {
      const v = entropy.pixels[i] / 255;
      if (v >= options.entropyThreshold) {
        color = blend(color, paletteColor(options.entropyPalette, v), options.entropyAlpha * v);
      }
    }

    rgba[j] = color[0];
    rgba[j + 1] = color[1];
    rgba[j + 2] = color[2];
    rgba[j + 3] = 255;
  }

  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.putImageData(new ImageData(rgba, image.width, image.height), 0, 0);
  return canvas.toDataURL("image/png");
}

function computeSliceMetrics(prediction: GrayImage, entropy: GrayImage, entropyThreshold: number): SliceMetrics {
  const total = prediction.pixels.length;
  let tumorPixels = 0;
  let hotspotPixels = 0;
  let entropyAccum = 0;

  for (let i = 0; i < total; i++) {
    const pred = prediction.pixels[i] / 255;
    const ent = entropy.pixels[i] / 255;
    if (pred >= 0.5) tumorPixels += 1;
    if (ent >= entropyThreshold) hotspotPixels += 1;
    entropyAccum += ent;
  }

  return {
    tumorPct: (tumorPixels / total) * 100,
    hotspotPct: (hotspotPixels / total) * 100,
    entropyMean: entropyAccum / total,
  };
}

export default function HomePage() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [caseId, setCaseId] = useState(DEFAULT_CASE_ID);
  const [imagePath, setImagePath] = useState("");
  const [predictionPath, setPredictionPath] = useState("");
  const [entropyPath, setEntropyPath] = useState("");
  const [imageChannel, setImageChannel] = useState(0);
  const [imageChannelAxis, setImageChannelAxis] = useState<ChannelAxis>("last");

  const [meta, setMeta] = useState<CaseMeta | null>(null);
  const [activeCaseId, setActiveCaseId] = useState(DEFAULT_CASE_ID);
  const [sliceIndex, setSliceIndex] = useState(0);

  const [displayMode, setDisplayMode] = useState<DisplayMode>("composite");
  const [showPrediction, setShowPrediction] = useState(true);
  const [showEntropy, setShowEntropy] = useState(true);
  const [predictionAlpha, setPredictionAlpha] = useState(0.45);
  const [entropyAlpha, setEntropyAlpha] = useState(0.55);
  const [predictionThreshold, setPredictionThreshold] = useState(0.3);
  const [entropyThreshold, setEntropyThreshold] = useState(0.5);
  const [predictionPalette, setPredictionPalette] = useState<Palette>("ember");
  const [entropyPalette, setEntropyPalette] = useState<Palette>("ocean");
  const [refreshKey, setRefreshKey] = useState(0);
  const [sliceStride3d, setSliceStride3d] = useState(4);
  const [pixelStride3d, setPixelStride3d] = useState(4);
  const [maxPoints3d, setMaxPoints3d] = useState(12000);
  const [imageThreshold3d, setImageThreshold3d] = useState(0);
  const [scatter3d, setScatter3d] = useState<Scatter3DPayload | null>(null);
  const [scatterLoading, setScatterLoading] = useState(false);
  const [scatterError, setScatterError] = useState<string | null>(null);

  const [sliceLoading, setSliceLoading] = useState(false);
  const [sliceError, setSliceError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("Idle");
  const [error, setError] = useState<string | null>(null);
  const [layers, setLayers] = useState<LayerImages | null>(null);
  const [layerUrls, setLayerUrls] = useState<LayerUrls | null>(null);

  const urlsRef = useRef<LayerUrls | null>(null);

  const sanitizedApiUrl = useMemo(() => normalizeApiUrl(apiUrl), [apiUrl]);

  const clearLayerUrls = () => {
    if (!urlsRef.current) return;
    URL.revokeObjectURL(urlsRef.current.image);
    URL.revokeObjectURL(urlsRef.current.prediction);
    URL.revokeObjectURL(urlsRef.current.entropy);
    urlsRef.current = null;
    setLayerUrls(null);
  };

  useEffect(() => {
    return () => clearLayerUrls();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const buildLayerPath = (layer: Layer, z: number) => buildCaseLayerPath(activeCaseId, layer, z, refreshKey);

  const onCheckApi = async () => {
    setError(null);
    try {
      const response = await proxyFetch(sanitizedApiUrl, "/health");
      if (!response.ok) {
        throw new Error(await readApiError(response));
      }
      const payload = (await response.json()) as { status?: string };
      setStatus(`API healthy (${payload.status ?? "ok"})`);
    } catch (err) {
      setStatus("API unavailable");
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  };

  const onRegisterCase = async () => {
    setError(null);
    try {
      const response = await proxyFetch(sanitizedApiUrl, "/cases/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          case_id: caseId,
          image_path: imagePath,
          prediction_path: predictionPath,
          entropy_path: entropyPath,
          image_channel: imageChannel,
          image_channel_axis: imageChannelAxis,
        }),
      });
      if (!response.ok) {
        throw new Error(await readApiError(response));
      }
      const payload = (await response.json()) as CaseMeta;
      setMeta(payload);
      setActiveCaseId(payload.case_id);
      setSliceIndex(Math.floor(payload.z_slices / 2));
      setRefreshKey((prev) => prev + 1);
      setScatter3d(null);
      setScatterError(null);
      setStatus(`Loaded case '${payload.case_id}'`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  };

  const onLoadExisting = async () => {
    setError(null);
    try {
      const response = await proxyFetch(sanitizedApiUrl, `/cases/${encodeURIComponent(caseId)}/meta`);
      if (!response.ok) {
        throw new Error(await readApiError(response));
      }
      const payload = (await response.json()) as CaseMeta;
      setMeta(payload);
      setActiveCaseId(payload.case_id);
      setSliceIndex(Math.floor(payload.z_slices / 2));
      setRefreshKey((prev) => prev + 1);
      setScatter3d(null);
      setScatterError(null);
      setStatus(`Attached to case '${payload.case_id}'`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    }
  };

  const onBuild3dScatter = async () => {
    if (!meta) return;
    setScatterLoading(true);
    setScatterError(null);
    setStatus("Building 3D scatter...");
    try {
      const params = new URLSearchParams({
        image_threshold: imageThreshold3d.toString(),
        slice_stride: String(sliceStride3d),
        pixel_stride: String(pixelStride3d),
        max_points: String(maxPoints3d),
        seed: "42",
      });
      const response = await proxyFetch(
        sanitizedApiUrl,
        `/cases/${encodeURIComponent(activeCaseId)}/scatter3d?${params.toString()}`,
      );
      if (!response.ok) {
        throw new Error(await readApiError(response));
      }
      const payload = (await response.json()) as Scatter3DPayload;
      setScatter3d(payload);
      setStatus(`3D scatter ready (${payload.counts.total.toLocaleString()} points)`);
    } catch (err) {
      setScatterError(err instanceof Error ? err.message : "Failed to build 3D scatter.");
    } finally {
      setScatterLoading(false);
    }
  };

  useEffect(() => {
    if (!meta) return;
    const maxSlice = meta.z_slices - 1;
    const z = clamp(sliceIndex, 0, maxSlice);
    if (z !== sliceIndex) {
      setSliceIndex(z);
      return;
    }

    const controller = new AbortController();
    const load = async () => {
      setSliceLoading(true);
      setSliceError(null);
      try {
        const [imageResult, predResult, entropyResult] = await Promise.all([
          fetchGrayImage(sanitizedApiUrl, buildLayerPath("image", z), controller.signal),
          fetchGrayImage(sanitizedApiUrl, buildLayerPath("prediction", z), controller.signal),
          fetchGrayImage(sanitizedApiUrl, buildLayerPath("entropy", z), controller.signal),
        ]);

        if (controller.signal.aborted) return;
        clearLayerUrls();

        const urls: LayerUrls = {
          image: URL.createObjectURL(imageResult.blob),
          prediction: URL.createObjectURL(predResult.blob),
          entropy: URL.createObjectURL(entropyResult.blob),
        };
        urlsRef.current = urls;
        setLayerUrls(urls);
        setLayers({
          image: imageResult.gray,
          prediction: predResult.gray,
          entropy: entropyResult.gray,
        });
      } catch (err) {
        if (!controller.signal.aborted) {
          setSliceError(err instanceof Error ? err.message : "Failed to load slice.");
        }
      } finally {
        if (!controller.signal.aborted) {
          setSliceLoading(false);
        }
      }
    };

    load();
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta, sliceIndex, refreshKey, activeCaseId, sanitizedApiUrl]);

  const metrics = useMemo(() => {
    if (!layers) return null;
    return computeSliceMetrics(layers.prediction, layers.entropy, entropyThreshold);
  }, [layers, entropyThreshold]);

  const compositeUrl = useMemo(() => {
    if (!layers) return "";
    return buildCompositeDataUrl(layers, {
      showPrediction,
      showEntropy,
      predictionAlpha,
      entropyAlpha,
      predictionThreshold,
      entropyThreshold,
      predictionPalette,
      entropyPalette,
    });
  }, [
    layers,
    showPrediction,
    showEntropy,
    predictionAlpha,
    entropyAlpha,
    predictionThreshold,
    entropyThreshold,
    predictionPalette,
    entropyPalette,
  ]);

  const maxSlice = meta ? meta.z_slices - 1 : 0;
  const scatterPlotData = useMemo<Data[]>(() => {
    if (!scatter3d) return [];
    const pointSize = scatter3d.point_size;
    return [
      {
        type: "scatter3d" as const,
        mode: "markers" as const,
        name: "Tissue",
        x: scatter3d.yellow.x,
        y: scatter3d.yellow.y,
        z: scatter3d.yellow.z,
        marker: {
          size: pointSize,
          color: "#facc15",
          opacity: 0.2,
        },
      },
      {
        type: "scatter3d" as const,
        mode: "markers" as const,
        name: "Prediction",
        x: scatter3d.red.x,
        y: scatter3d.red.y,
        z: scatter3d.red.z,
        marker: {
          size: pointSize + 0.35,
          color: "#ef4444",
          opacity: 0.82,
        },
      },
    ];
  }, [scatter3d]);

  const scatterLayout = useMemo<Partial<Layout>>(
    () => ({
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: { l: 0, r: 0, t: 0, b: 0 },
      scene: {
        xaxis: { visible: false },
        yaxis: { visible: false },
        zaxis: { visible: false },
        aspectmode: "cube",
        bgcolor: "rgba(8,19,29,1)",
      },
      legend: {
        orientation: "h" as const,
        x: 0,
        y: 1.02,
        font: { color: "#e8eef2" },
      },
    }),
    [],
  );

  return (
    <div className="page">
      <aside className="sidebar">
        <div className="sidebar-card">
          <h2>NeuroMap Console</h2>
          <p className="subtle">Connect to FastAPI and register a case.</p>
        </div>

        <label className="field">
          <span>API URL</span>
          <input value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} placeholder={DEFAULT_API_URL} />
        </label>
        <button className="button button-outline" onClick={onCheckApi}>
          Check API
        </button>

        <hr className="divider" />

        <label className="field">
          <span>Case ID</span>
          <input value={caseId} onChange={(e) => setCaseId(e.target.value)} />
        </label>
        <label className="field">
          <span>MRI volume path</span>
          <input value={imagePath} onChange={(e) => setImagePath(e.target.value)} />
        </label>
        <label className="field">
          <span>Prediction path</span>
          <input value={predictionPath} onChange={(e) => setPredictionPath(e.target.value)} />
        </label>
        <label className="field">
          <span>Entropy path</span>
          <input value={entropyPath} onChange={(e) => setEntropyPath(e.target.value)} />
        </label>

        <div className="row">
          <label className="field">
            <span>Channel index</span>
            <input
              type="number"
              value={imageChannel}
              onChange={(e) => setImageChannel(Number(e.target.value))}
              min={0}
            />
          </label>
          <label className="field">
            <span>Channel axis</span>
            <select value={imageChannelAxis} onChange={(e) => setImageChannelAxis(e.target.value as ChannelAxis)}>
              <option value="last">last</option>
              <option value="first">first</option>
            </select>
          </label>
        </div>

        <button className="button" onClick={onRegisterCase}>
          Register Case
        </button>
        <button className="button button-outline" onClick={onLoadExisting}>
          Load Existing Case
        </button>
        <button
          className="button button-outline"
          onClick={() => {
            setRefreshKey((prev) => prev + 1);
            setStatus("Slice cache refreshed");
          }}
        >
          Refresh Slices
        </button>

        <div className="status-wrap">
          <div className="status-ok">{status}</div>
          {error ? <div className="status-error">{error}</div> : null}
        </div>
      </aside>

      <main className="main">
        <section className="hero">
          <p className="hero-badge">3D Volumetric Segmentation + UQ</p>
          <h1>NeuroMap Radiology Workbench</h1>
          <p>
            Scroll Z-slices, inspect tumor boundaries, and surface entropy hotspots where model confidence is weak.
          </p>
        </section>

        {!meta ? (
          <section className="empty">Register or load a case from the left panel to begin.</section>
        ) : (
          <>
            <section className="toolbar">
              <button className="button button-outline" onClick={() => setSliceIndex((z) => clamp(z - 1, 0, maxSlice))}>
                Prev Slice
              </button>
              <input
                type="range"
                min={0}
                max={maxSlice}
                value={sliceIndex}
                onChange={(e) => setSliceIndex(Number(e.target.value))}
              />
              <button
                className="button button-outline"
                onClick={() => setSliceIndex((z) => clamp(z + 1, 0, maxSlice))}
              >
                Next Slice
              </button>
            </section>

            <section className="stats">
              <article className="stat-card">
                <h3>Case</h3>
                <p>{activeCaseId}</p>
              </article>
              <article className="stat-card">
                <h3>Shape</h3>
                <p className="mono">{meta.shape.join(" x ")}</p>
              </article>
              <article className="stat-card">
                <h3>Slice</h3>
                <p className="mono">
                  {sliceIndex} / {maxSlice}
                </p>
              </article>
              <article className="stat-card">
                <h3>Tumor Pixels</h3>
                <p>{metrics ? `${metrics.tumorPct.toFixed(2)}%` : "--"}</p>
              </article>
              <article className="stat-card">
                <h3>Hotspot Pixels</h3>
                <p>{metrics ? `${metrics.hotspotPct.toFixed(2)}%` : "--"}</p>
              </article>
              <article className="stat-card">
                <h3>Entropy Mean</h3>
                <p>{metrics ? metrics.entropyMean.toFixed(3) : "--"}</p>
              </article>
            </section>

            <section className="controls-grid">
              <label className="field">
                <span>Display Mode</span>
                <select value={displayMode} onChange={(e) => setDisplayMode(e.target.value as DisplayMode)}>
                  <option value="composite">Composite</option>
                  <option value="sideBySide">Side by side</option>
                  <option value="imageOnly">Image only</option>
                </select>
              </label>
              <label className="field checkbox">
                <input
                  type="checkbox"
                  checked={showPrediction}
                  onChange={(e) => setShowPrediction(e.target.checked)}
                />
                <span>Prediction overlay</span>
              </label>
              <label className="field checkbox">
                <input type="checkbox" checked={showEntropy} onChange={(e) => setShowEntropy(e.target.checked)} />
                <span>Uncertainty overlay</span>
              </label>
              <label className="field">
                <span>Prediction alpha ({predictionAlpha.toFixed(2)})</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={predictionAlpha}
                  onChange={(e) => setPredictionAlpha(Number(e.target.value))}
                />
              </label>
              <label className="field">
                <span>Uncertainty alpha ({entropyAlpha.toFixed(2)})</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={entropyAlpha}
                  onChange={(e) => setEntropyAlpha(Number(e.target.value))}
                />
              </label>
              <label className="field">
                <span>Prediction threshold ({predictionThreshold.toFixed(2)})</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={predictionThreshold}
                  onChange={(e) => setPredictionThreshold(Number(e.target.value))}
                />
              </label>
              <label className="field">
                <span>Uncertainty threshold ({entropyThreshold.toFixed(2)})</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={entropyThreshold}
                  onChange={(e) => setEntropyThreshold(Number(e.target.value))}
                />
              </label>
              <label className="field">
                <span>Prediction palette</span>
                <select value={predictionPalette} onChange={(e) => setPredictionPalette(e.target.value as Palette)}>
                  <option value="ember">Ember</option>
                  <option value="mint">Mint</option>
                  <option value="ocean">Ocean</option>
                </select>
              </label>
              <label className="field">
                <span>Uncertainty palette</span>
                <select value={entropyPalette} onChange={(e) => setEntropyPalette(e.target.value as Palette)}>
                  <option value="ocean">Ocean</option>
                  <option value="ember">Ember</option>
                  <option value="mint">Mint</option>
                </select>
              </label>
            </section>

            <section className="scatter3d-panel">
              <div className="scatter3d-header">
                <h3>3D Scatter (Plotly)</h3>
                <p>Black/background voxels are skipped. Tissue is yellow, prediction voxels are red.</p>
              </div>
              <div className="scatter3d-controls">
                <label className="field">
                  <span>Slice stride</span>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={sliceStride3d}
                    onChange={(e) => setSliceStride3d(clamp(Number(e.target.value) || 1, 1, 10))}
                  />
                </label>
                <label className="field">
                  <span>Pixel stride</span>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={pixelStride3d}
                    onChange={(e) => setPixelStride3d(clamp(Number(e.target.value) || 1, 1, 10))}
                  />
                </label>
                <label className="field">
                  <span>Max points</span>
                  <input
                    type="number"
                    min={1000}
                    max={500000}
                    step={1000}
                    value={maxPoints3d}
                    onChange={(e) => setMaxPoints3d(clamp(Number(e.target.value) || 1000, 1000, 500000))}
                  />
                </label>
                <label className="field">
                  <span>Image threshold</span>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    value={imageThreshold3d}
                    onChange={(e) => setImageThreshold3d(Math.max(0, Number(e.target.value) || 0))}
                  />
                </label>
                <button className="button" onClick={onBuild3dScatter} disabled={scatterLoading}>
                  {scatterLoading ? "Building..." : "Build 3D Scatter"}
                </button>
                <button
                  className="button button-outline"
                  onClick={() => {
                    setSliceStride3d(5);
                    setPixelStride3d(5);
                    setMaxPoints3d(8000);
                    setStatus("Applied low-density 3D preset");
                  }}
                >
                  Low Density Preset
                </button>
                <button
                  className="button button-outline"
                  onClick={() => {
                    setScatter3d(null);
                    setScatterError(null);
                    setStatus("3D scatter cleared");
                  }}
                >
                  Clear 3D
                </button>
              </div>
              {scatterError ? <div className="status-error">{scatterError}</div> : null}
              {scatter3d ? (
                <div className="scatter3d-meta">
                  <span>Total: {scatter3d.counts.total.toLocaleString()}</span>
                  <span>Yellow: {scatter3d.counts.yellow.toLocaleString()}</span>
                  <span>Red: {scatter3d.counts.red.toLocaleString()}</span>
                </div>
              ) : null}
              <div className="scatter3d-plot-wrap">
                {scatter3d ? (
                  <Plot
                    data={scatterPlotData}
                    layout={scatterLayout}
                    config={{ responsive: true, displaylogo: false }}
                    style={{ width: "100%", height: "520px" }}
                    useResizeHandler
                  />
                ) : (
                  <div className="empty">Build 3D scatter to render the volume.</div>
                )}
              </div>
            </section>

            <section className="viewer">
              {sliceLoading ? <div className="empty">Loading slice...</div> : null}
              {sliceError ? <div className="status-error">{sliceError}</div> : null}

              {!sliceLoading && !sliceError && layerUrls && (
                <>
                  {displayMode === "composite" && compositeUrl ? (
                    <img className="scan" src={compositeUrl} alt={`Composite slice ${sliceIndex}`} />
                  ) : null}

                  {displayMode === "imageOnly" ? (
                    <img className="scan" src={layerUrls.image} alt={`MRI slice ${sliceIndex}`} />
                  ) : null}

                  {displayMode === "sideBySide" ? (
                    <div className="side-grid">
                      <figure>
                        <img src={layerUrls.image} alt={`MRI slice ${sliceIndex}`} />
                        <figcaption>MRI</figcaption>
                      </figure>
                      <figure>
                        <img src={layerUrls.prediction} alt={`Prediction slice ${sliceIndex}`} />
                        <figcaption>Prediction</figcaption>
                      </figure>
                      <figure>
                        <img src={layerUrls.entropy} alt={`Entropy slice ${sliceIndex}`} />
                        <figcaption>Entropy</figcaption>
                      </figure>
                    </div>
                  ) : null}
                </>
              )}
            </section>
          </>
        )}
      </main>
    </div>
  );
}
