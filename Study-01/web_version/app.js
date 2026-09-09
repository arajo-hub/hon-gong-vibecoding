// Ink Reader (web) — draws a digit and recognizes it with a network
// trained by ../train_mnist.py. Weights are fetched from ../weights.json
// at load time; the forward pass below must stay numerically consistent
// with MLP.forward in train_mnist.py.

let WEIGHTS = null;

function forward(x) {
  const { w1, b1, w2, b2, hidden_size } = WEIGHTS;
  const h = new Float32Array(hidden_size);
  for (let j = 0; j < hidden_size; j++) {
    let s = b1[j];
    for (let i = 0; i < 784; i++) s += x[i] * w1[i][j];
    h[j] = s > 0 ? s : 0;
  }
  const out = new Float32Array(10);
  for (let k = 0; k < 10; k++) {
    let s = b2[k];
    for (let j = 0; j < hidden_size; j++) s += h[j] * w2[j][k];
    out[k] = s;
  }
  let max = -Infinity;
  for (let k = 0; k < 10; k++) if (out[k] > max) max = out[k];
  let sum = 0;
  const exps = new Float32Array(10);
  for (let k = 0; k < 10; k++) { exps[k] = Math.exp(out[k] - max); sum += exps[k]; }
  for (let k = 0; k < 10; k++) exps[k] /= sum;
  return exps;
}

// ---- Canvas setup ----
const canvas = document.getElementById("pad");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const DPR = Math.min(window.devicePixelRatio || 1, 3);
const LOGICAL_SIZE = 280;

function sizeCanvas() {
  canvas.width = LOGICAL_SIZE * DPR;
  canvas.height = LOGICAL_SIZE * DPR;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = 18;
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim() || "#211f2b";
}
sizeCanvas();

function clearCanvas() {
  ctx.clearRect(0, 0, LOGICAL_SIZE, LOGICAL_SIZE);
}

// ---- Digit preprocessing: crop to ink bounding box, resize to 28x28 ----
function extractVector() {
  const w = canvas.width, h = canvas.height;
  const data = ctx.getImageData(0, 0, w, h).data;
  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const a = data[(y * w + x) * 4 + 3];
      if (a > 10) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;

  const bboxW = maxX - minX + 1;
  const bboxH = maxY - minY + 1;
  const size = Math.max(bboxW, bboxH);
  const pad = size * 0.28;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const half = size / 2 + pad;

  const crop = document.createElement("canvas");
  crop.width = 28;
  crop.height = 28;
  const cctx = crop.getContext("2d");
  cctx.imageSmoothingEnabled = true;
  cctx.imageSmoothingQuality = "high";
  cctx.drawImage(canvas, cx - half, cy - half, half * 2, half * 2, 0, 0, 28, 28);

  const pixels = cctx.getImageData(0, 0, 28, 28).data;
  const vec = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    const idx = i * 4;
    const alpha = pixels[idx + 3] / 255;
    const lum = (pixels[idx] * 0.299 + pixels[idx + 1] * 0.587 + pixels[idx + 2] * 0.114) / 255;
    // Ink strokes are opaque + dark; MNIST format is bright digit on black background.
    vec[i] = alpha * (1 - lum);
  }
  return vec;
}

// ---- Result rendering ----
const predictedDigitEl = document.getElementById("predictedDigit");
const confidenceEl = document.getElementById("confidenceValue");
const barsEl = document.getElementById("bars");
const clearBtn = document.getElementById("clear");
const hintEl = document.getElementById("hint");

for (let d = 0; d < 10; d++) {
  const row = document.createElement("div");
  row.className = "bar-row";
  row.id = "bar-row-" + d;
  row.innerHTML =
    '<span class="bar-label">' + d + '</span>' +
    '<span class="bar-track"><span class="bar-fill" id="bar-fill-' + d + '"></span></span>' +
    '<span class="bar-pct" id="bar-pct-' + d + '">0%</span>';
  barsEl.appendChild(row);
}

function showEmptyState() {
  predictedDigitEl.textContent = "–";
  confidenceEl.textContent = "–";
  for (let d = 0; d < 10; d++) {
    document.getElementById("bar-row-" + d).classList.remove("top");
    document.getElementById("bar-fill-" + d).style.width = "0%";
    document.getElementById("bar-pct-" + d).textContent = "0%";
  }
}

function runPrediction() {
  if (!WEIGHTS) return;
  const vec = extractVector();
  if (!vec) { showEmptyState(); return; }
  const probs = forward(vec);
  let best = 0;
  for (let d = 1; d < 10; d++) if (probs[d] > probs[best]) best = d;

  predictedDigitEl.textContent = String(best);
  confidenceEl.textContent = (probs[best] * 100).toFixed(1) + "%";

  for (let d = 0; d < 10; d++) {
    const pct = probs[d] * 100;
    document.getElementById("bar-row-" + d).classList.toggle("top", d === best);
    document.getElementById("bar-fill-" + d).style.width = pct.toFixed(1) + "%";
    document.getElementById("bar-pct-" + d).textContent = pct.toFixed(1) + "%";
  }
}

// ---- Drawing interaction ----
let drawing = false;
let lastPoint = null;
let isExample = true;
const exampleTag = document.getElementById("exampleTag");
let lastPredictAt = 0;

function pointFromEvent(evt) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (evt.clientX - rect.left) * (LOGICAL_SIZE / rect.width),
    y: (evt.clientY - rect.top) * (LOGICAL_SIZE / rect.height),
  };
}

function startStroke(evt) {
  if (isExample) {
    clearCanvas();
    isExample = false;
    exampleTag.hidden = true;
  }
  drawing = true;
  lastPoint = pointFromEvent(evt);
  canvas.setPointerCapture(evt.pointerId);
}

function moveStroke(evt) {
  if (!drawing) return;
  const p = pointFromEvent(evt);
  ctx.beginPath();
  ctx.moveTo(lastPoint.x, lastPoint.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  lastPoint = p;

  const now = performance.now();
  if (now - lastPredictAt > 120) {
    lastPredictAt = now;
    runPrediction();
  }
}

function endStroke() {
  if (!drawing) return;
  drawing = false;
  runPrediction();
}

canvas.addEventListener("pointerdown", startStroke);
canvas.addEventListener("pointermove", moveStroke);
canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointerleave", endStroke);

clearBtn.addEventListener("click", () => {
  clearCanvas();
  isExample = false;
  exampleTag.hidden = true;
  showEmptyState();
});

// ---- Draw an initial example stroke ("1") so the tool opens already working ----
function drawExampleDigit() {
  ctx.beginPath();
  ctx.moveTo(118, 62);
  ctx.lineTo(150, 46);
  ctx.lineTo(150, 218);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(112, 218);
  ctx.lineTo(196, 218);
  ctx.stroke();
}

// ---- Load weights, then draw the example and enable interaction ----
clearBtn.disabled = true;
hintEl.textContent = "Loading model…";

fetch("../weights.json")
  .then((res) => {
    if (!res.ok) throw new Error("HTTP " + res.status);
    return res.json();
  })
  .then((weights) => {
    WEIGHTS = weights;
    clearBtn.disabled = false;
    hintEl.textContent = "Draw over it to try your own";
    drawExampleDigit();
    runPrediction();
  })
  .catch((err) => {
    hintEl.textContent = "Could not load weights.json — serve this folder's parent over HTTP (see CLAUDE.md).";
    console.error(err);
  });
