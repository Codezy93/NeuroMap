const severityFilter = document.getElementById("severityFilter");
const insightList = document.getElementById("insightList");

const zSlider = document.getElementById("zSlider");
const zValue = document.getElementById("zValue");
const zMax = document.getElementById("zMax");
const prevSlice = document.getElementById("prevSlice");
const nextSlice = document.getElementById("nextSlice");
const displayMode = document.getElementById("displayMode");
const togglePrediction = document.getElementById("togglePrediction");
const toggleEntropy = document.getElementById("toggleEntropy");
const toggleVariance = document.getElementById("toggleVariance");
const predictionAlpha = document.getElementById("predictionAlpha");
const entropyAlpha = document.getElementById("entropyAlpha");
const varianceAlpha = document.getElementById("varianceAlpha");
const predictionThreshold = document.getElementById("predictionThreshold");
const entropyThreshold = document.getElementById("entropyThreshold");
const varianceThreshold = document.getElementById("varianceThreshold");
const predictionAlphaValue = document.getElementById("predictionAlphaValue");
const entropyAlphaValue = document.getElementById("entropyAlphaValue");
const varianceAlphaValue = document.getElementById("varianceAlphaValue");
const predictionThresholdValue = document.getElementById("predictionThresholdValue");
const entropyThresholdValue = document.getElementById("entropyThresholdValue");
const varianceThresholdValue = document.getElementById("varianceThresholdValue");
const metricTumor = document.getElementById("metricTumor");
const metricHotspot = document.getElementById("metricHotspot");
const metricEntropy = document.getElementById("metricEntropy");
const compositeCanvas = document.getElementById("compositeCanvas");
const sideGrid = document.getElementById("sideGrid");
const layerMeanProb = document.getElementById("layerMeanProb");
const layerPrediction = document.getElementById("layerPrediction");
const layerEntropy = document.getElementById("layerEntropy");
const layerVariance = document.getElementById("layerVariance");
const scatterPreset = document.getElementById("scatterPreset");
const loadScatter = document.getElementById("loadScatter");
const scatterCounts = document.getElementById("scatterCounts");
const scatterStatus = document.getElementById("scatterStatus");
const scatterPlot = document.getElementById("scatterPlot");

const overlayColors = {
  prediction: [239, 68, 68],
  entropy: [56, 189, 248],
  variance: [250, 204, 21],
};

const state = {
  manifest: null,
  z: 77,
  maxZ: 154,
  currentSlices: null,
  renderToken: 0,
};

const imageCache = new Map();
const offscreenCanvas = document.createElement("canvas");
const offscreenCtx = offscreenCanvas.getContext("2d");

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function padZ(z) {
  return String(z).padStart(3, "0");
}

function filterInsights() {
  if (!severityFilter || !insightList) return;
  const selected = severityFilter.value;
  const cards = insightList.querySelectorAll(".insight-card");
  cards.forEach((card) => {
    const severity = card.getAttribute("data-severity");
    const show = selected === "all" || severity === selected;
    card.classList.toggle("hidden", !show);
  });
}

function slicePath(layer, z) {
  return `assets/slices/${layer}/z${padZ(z)}.png`;
}

function updateControlLabels() {
  if (predictionAlphaValue && predictionAlpha) predictionAlphaValue.textContent = Number(predictionAlpha.value).toFixed(2);
  if (entropyAlphaValue && entropyAlpha) entropyAlphaValue.textContent = Number(entropyAlpha.value).toFixed(2);
  if (varianceAlphaValue && varianceAlpha) varianceAlphaValue.textContent = Number(varianceAlpha.value).toFixed(2);
  if (predictionThresholdValue && predictionThreshold) predictionThresholdValue.textContent = Number(predictionThreshold.value).toFixed(2);
  if (entropyThresholdValue && entropyThreshold) entropyThresholdValue.textContent = Number(entropyThreshold.value).toFixed(2);
  if (varianceThresholdValue && varianceThreshold) varianceThresholdValue.textContent = Number(varianceThreshold.value).toFixed(2);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
}

async function getGraySlice(layer, z) {
  const key = `${layer}:${z}`;
  if (imageCache.has(key)) return imageCache.get(key);

  if (!offscreenCtx) throw new Error("Canvas context unavailable.");

  const src = slicePath(layer, z);
  const img = await loadImage(src);
  offscreenCanvas.width = img.naturalWidth || img.width;
  offscreenCanvas.height = img.naturalHeight || img.height;
  offscreenCtx.clearRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);
  offscreenCtx.drawImage(img, 0, 0);
  const rgba = offscreenCtx.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height).data;
  const gray = new Uint8ClampedArray(offscreenCanvas.width * offscreenCanvas.height);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    gray[i] = rgba[j];
  }

  const payload = {
    width: offscreenCanvas.width,
    height: offscreenCanvas.height,
    gray,
    src,
  };
  imageCache.set(key, payload);
  return payload;
}

function mix(base, overlay, alpha) {
  return Math.round(base * (1 - alpha) + overlay * alpha);
}

function drawComposite() {
  if (!compositeCanvas || !state.currentSlices) return;
  const ctx = compositeCanvas.getContext("2d");
  if (!ctx) return;

  const mean = state.currentSlices.mean_prob;
  const pred = state.currentSlices.prediction;
  const ent = state.currentSlices.entropy;
  const vari = state.currentSlices.variance;

  const total = mean.gray.length;
  const rgba = new Uint8ClampedArray(total * 4);

  const predEnabled = togglePrediction ? togglePrediction.checked : true;
  const entEnabled = toggleEntropy ? toggleEntropy.checked : true;
  const varEnabled = toggleVariance ? toggleVariance.checked : false;

  const predAlpha = predictionAlpha ? Number(predictionAlpha.value) : 0.5;
  const entAlpha = entropyAlpha ? Number(entropyAlpha.value) : 0.55;
  const varAlpha = varianceAlpha ? Number(varianceAlpha.value) : 0.35;
  const predThresh = predictionThreshold ? Number(predictionThreshold.value) : 0.3;
  const entThresh = entropyThreshold ? Number(entropyThreshold.value) : 0.5;
  const varThresh = varianceThreshold ? Number(varianceThreshold.value) : 0.35;

  for (let i = 0, j = 0; i < total; i++, j += 4) {
    const base = mean.gray[i] / 255;
    let r = mean.gray[i];
    let g = mean.gray[i];
    let b = mean.gray[i];

    const p = pred.gray[i] / 255;
    if (predEnabled && p >= predThresh) {
      const a = predAlpha * p;
      r = mix(r, overlayColors.prediction[0], a);
      g = mix(g, overlayColors.prediction[1], a);
      b = mix(b, overlayColors.prediction[2], a);
    }

    const e = ent.gray[i] / 255;
    if (entEnabled && e >= entThresh) {
      const a = entAlpha * e;
      r = mix(r, overlayColors.entropy[0], a);
      g = mix(g, overlayColors.entropy[1], a);
      b = mix(b, overlayColors.entropy[2], a);
    }

    const v = vari.gray[i] / 255;
    if (varEnabled && v >= varThresh) {
      const a = varAlpha * v;
      r = mix(r, overlayColors.variance[0], a);
      g = mix(g, overlayColors.variance[1], a);
      b = mix(b, overlayColors.variance[2], a);
    }

    const brightnessBoost = base < 0.14 ? 1.07 : 1.0;
    rgba[j] = clamp(Math.round(r * brightnessBoost), 0, 255);
    rgba[j + 1] = clamp(Math.round(g * brightnessBoost), 0, 255);
    rgba[j + 2] = clamp(Math.round(b * brightnessBoost), 0, 255);
    rgba[j + 3] = 255;
  }

  compositeCanvas.width = mean.width;
  compositeCanvas.height = mean.height;
  ctx.putImageData(new ImageData(rgba, mean.width, mean.height), 0, 0);
}

function updateViewerMode() {
  if (!displayMode || !compositeCanvas || !sideGrid) return;
  const mode = displayMode.value;
  if (mode === "sideBySide") {
    compositeCanvas.style.display = "none";
    sideGrid.style.display = "grid";
  } else {
    compositeCanvas.style.display = "block";
    sideGrid.style.display = mode === "imageOnly" ? "none" : "none";
  }
}

function updateSliceMetrics() {
  if (!state.currentSlices || !metricTumor || !metricHotspot || !metricEntropy) return;
  const pred = state.currentSlices.prediction.gray;
  const ent = state.currentSlices.entropy.gray;
  const total = pred.length;
  const entThresh = entropyThreshold ? Number(entropyThreshold.value) : 0.5;

  let tumor = 0;
  let hotspot = 0;
  let entropyAccum = 0;

  for (let i = 0; i < total; i++) {
    const p = pred[i] / 255;
    const e = ent[i] / 255;
    if (p >= 0.5) tumor += 1;
    if (e >= entThresh) hotspot += 1;
    entropyAccum += e;
  }

  metricTumor.textContent = `${((tumor / total) * 100).toFixed(2)}%`;
  metricHotspot.textContent = `${((hotspot / total) * 100).toFixed(2)}%`;
  metricEntropy.textContent = (entropyAccum / total).toFixed(3);
}

async function renderSlice() {
  if (!zSlider || !zValue) return;
  const token = ++state.renderToken;
  const z = clamp(Number(zSlider.value), 0, state.maxZ);
  state.z = z;
  zSlider.value = String(z);
  zValue.textContent = String(z);

  const [meanSlice, predSlice, entSlice, varSlice] = await Promise.all([
    getGraySlice("mean_prob", z),
    getGraySlice("prediction", z),
    getGraySlice("entropy", z),
    getGraySlice("variance", z),
  ]);
  if (token !== state.renderToken) return;

  state.currentSlices = {
    mean_prob: meanSlice,
    prediction: predSlice,
    entropy: entSlice,
    variance: varSlice,
  };

  if (layerMeanProb) layerMeanProb.src = meanSlice.src;
  if (layerPrediction) layerPrediction.src = predSlice.src;
  if (layerEntropy) layerEntropy.src = entSlice.src;
  if (layerVariance) layerVariance.src = varSlice.src;

  updateSliceMetrics();
  updateViewerMode();

  if (displayMode && displayMode.value === "imageOnly") {
    const ctx = compositeCanvas ? compositeCanvas.getContext("2d") : null;
    if (ctx && compositeCanvas) {
      compositeCanvas.width = meanSlice.width;
      compositeCanvas.height = meanSlice.height;
      const rgba = new Uint8ClampedArray(meanSlice.gray.length * 4);
      for (let i = 0, j = 0; i < meanSlice.gray.length; i++, j += 4) {
        const v = meanSlice.gray[i];
        rgba[j] = v;
        rgba[j + 1] = v;
        rgba[j + 2] = v;
        rgba[j + 3] = 255;
      }
      ctx.putImageData(new ImageData(rgba, meanSlice.width, meanSlice.height), 0, 0);
    }
    return;
  }

  drawComposite();
}

async function loadManifest() {
  const response = await fetch("assets/manifest.json", { cache: "no-store" });
  if (!response.ok) throw new Error("Could not load assets/manifest.json");
  state.manifest = await response.json();
  state.maxZ = Math.max(0, Number(state.manifest.zSlices || 155) - 1);
  state.z = Math.floor(state.maxZ / 2);
}

function buildScatterLayout() {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 0, r: 0, b: 0, t: 0 },
    scene: {
      xaxis: { visible: false },
      yaxis: { visible: false },
      zaxis: { visible: false },
      bgcolor: "rgba(6,16,24,1)",
      aspectmode: "data",
    },
    legend: {
      orientation: "h",
      x: 0,
      y: 1.04,
      font: { color: "#dcefff" },
    },
  };
}

async function renderScatter() {
  if (!scatterPreset || !scatterCounts || !scatterStatus || !scatterPlot) return;
  const preset = scatterPreset.value;
  scatterStatus.textContent = "Loading preset...";

  const response = await fetch(`assets/scatter3d_${preset}.json`, { cache: "no-store" });
  if (!response.ok) {
    scatterStatus.textContent = "Failed to load 3D preset.";
    return;
  }

  const payload = await response.json();
  scatterCounts.textContent = `Points: ${payload.counts.total.toLocaleString()} (Y: ${payload.counts.yellow.toLocaleString()} | R: ${payload.counts.red.toLocaleString()})`;

  if (!window.Plotly) {
    scatterStatus.textContent = "Plotly did not load. Connect to the internet to render the 3D graph.";
    return;
  }

  const data = [
    {
      type: "scatter3d",
      mode: "markers",
      name: "Tissue",
      x: payload.yellow.x,
      y: payload.yellow.y,
      z: payload.yellow.z,
      marker: {
        size: payload.point_size + 0.9,
        color: "#facc15",
        opacity: 0.62,
      },
    },
    {
      type: "scatter3d",
      mode: "markers",
      name: "Prediction",
      x: payload.red.x,
      y: payload.red.y,
      z: payload.red.z,
      marker: {
        size: payload.point_size + 0.15,
        color: "#ef4444",
        opacity: 0.82,
      },
    },
  ];

  window.Plotly.react(scatterPlot, data, buildScatterLayout(), {
    displaylogo: false,
    responsive: true,
  });

  const imageThreshold = Number(payload.params.image_threshold ?? payload.params.white_threshold ?? 0);
  scatterStatus.textContent = `Preset: ${payload.preset} | source=${payload.params.tissue_source ?? "unknown"} | slice_stride=${payload.params.slice_stride}, pixel_stride=${payload.params.pixel_stride}, image_threshold=${imageThreshold.toFixed(3)}`;
}

function wireEvents() {
  if (severityFilter) severityFilter.addEventListener("change", filterInsights);

  if (zSlider) {
    zSlider.addEventListener("input", () => {
      renderSlice().catch((error) => {
        console.error(error);
      });
    });
  }

  if (prevSlice) {
    prevSlice.addEventListener("click", () => {
      if (!zSlider) return;
      zSlider.value = String(clamp(Number(zSlider.value) - 1, 0, state.maxZ));
      renderSlice().catch((error) => console.error(error));
    });
  }

  if (nextSlice) {
    nextSlice.addEventListener("click", () => {
      if (!zSlider) return;
      zSlider.value = String(clamp(Number(zSlider.value) + 1, 0, state.maxZ));
      renderSlice().catch((error) => console.error(error));
    });
  }

  const compositeTriggerIds = [
    displayMode,
    togglePrediction,
    toggleEntropy,
    toggleVariance,
    predictionAlpha,
    entropyAlpha,
    varianceAlpha,
    predictionThreshold,
    entropyThreshold,
    varianceThreshold,
  ];

  compositeTriggerIds.forEach((el) => {
    if (!el) return;
    el.addEventListener("input", () => {
      updateControlLabels();
      updateSliceMetrics();
      updateViewerMode();
      if (displayMode && displayMode.value === "imageOnly") {
        renderSlice().catch((error) => console.error(error));
      } else {
        drawComposite();
      }
    });
    el.addEventListener("change", () => {
      updateControlLabels();
      updateSliceMetrics();
      updateViewerMode();
      if (displayMode && displayMode.value === "imageOnly") {
        renderSlice().catch((error) => console.error(error));
      } else {
        drawComposite();
      }
    });
  });

  if (loadScatter) {
    loadScatter.addEventListener("click", () => {
      renderScatter().catch((error) => {
        if (scatterStatus) scatterStatus.textContent = `3D graph error: ${error.message}`;
      });
    });
  }

  if (scatterPreset) {
    scatterPreset.addEventListener("change", () => {
      renderScatter().catch((error) => {
        if (scatterStatus) scatterStatus.textContent = `3D graph error: ${error.message}`;
      });
    });
  }
}

async function boot() {
  filterInsights();
  wireEvents();
  updateControlLabels();

  try {
    await loadManifest();
    if (zSlider) {
      zSlider.max = String(state.maxZ);
      zSlider.value = String(state.z);
    }
    if (zMax) zMax.textContent = String(state.maxZ);
    if (zValue) zValue.textContent = String(state.z);
    await renderSlice();
    await renderScatter();
  } catch (error) {
    console.error(error);
    if (scatterStatus) {
      scatterStatus.textContent = `Initialization error: ${error.message}`;
    }
  }
}

boot();
