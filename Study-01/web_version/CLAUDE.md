# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The browser version of the Study-01 digit recognizer: a static page (`index.html` + `style.css` + `app.js`) with a drawing canvas. It fetches `../weights.json` at load time and runs the forward pass entirely client-side in vanilla JS — no build step, no server-side inference.

## Commands

Browsers block `fetch()` of local files over `file://`, so this must be served over HTTP — and served from **Study-01** (the parent directory), not from inside `web_version/`, so that the page's `../weights.json` request resolves:

```bash
cd .. && python3 -m http.server 8000
# then open http://localhost:8000/web_version/
```

There is no build step, bundler, or test suite — edit `index.html` / `style.css` / `app.js` directly and reload.

## Architecture

- `app.js` reimplements the same 784 → 128 (ReLU) → 10 (softmax) forward pass as `../train_mnist.py`'s `MLP.forward` and `../desktop_version/model.py`'s `forward`. If the network architecture or `weights.json`'s shape ever changes, update all three in lockstep — see the root `CLAUDE.md` for the array layout (`w1` is `[input][hidden]`, `w2` is `[hidden][output]`).
- Drawing pipeline (`extractVector` in `app.js`): the canvas is drawn at `devicePixelRatio` resolution for crisp strokes, then on each prediction the ink's bounding box is found, padded ~28%, and redrawn into an offscreen 28x28 canvas via `drawImage` (which does the resize). This crop-to-bounding-box step is what makes off-center or differently-scaled strokes classify well — skipping it and just downscaling the full canvas measurably hurts accuracy.
- The page ships with a pre-drawn example digit ("1") that's already recognized on load, so it never opens to an empty, unproven state. The first real pointer stroke clears it.
- Prediction re-runs continuously while drawing (throttled to ~120ms via `lastPredictAt`), not just on pointer-up, so the result panel feels live.
- No external JS dependencies — the forward pass is small enough (784x128 + 128x10 multiply-adds) to run in plain loops well within a frame budget.
