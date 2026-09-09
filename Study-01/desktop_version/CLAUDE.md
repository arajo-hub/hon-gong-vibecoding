# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The desktop version of the Study-01 digit recognizer: a native Tkinter app (`draw_app.py`) with a drawing canvas, plus a CLI (`predict.py`) that classifies an existing image file. Both load weights from `../weights.json` and share their forward pass / preprocessing through `model.py`.

## Commands

Requires the venv at `../.venv` to have Tkinter support (the base Homebrew Python does not ship `_tkinter` — it was added here via `brew install python-tk@3.13`, which must be installed for whichever Python built `../.venv` before Tkinter will import).

```bash
# Draw and recognize live
../.venv/bin/python draw_app.py

# Recognize a digit in an existing image file
../.venv/bin/python predict.py path/to/digit.png
```

There is no build step or test suite. `python -c "import tkinter; tkinter.Tk().destroy()"` is a quick way to confirm Tkinter itself is working before debugging app-level issues.

## Architecture

- `model.py` is the shared core: `load_weights()`, `forward()` (784 → 128 ReLU → 10 softmax, must stay numerically consistent with `MLP.forward` in `../train_mnist.py` and the reimplementation in `../web_version/app.js`), and `digit_image_to_vector()` — crops a drawing to its ink bounding box (padded ~28%), resizes to 28x28, and inverts it to MNIST's bright-digit-on-black format. Both `predict.py` and `draw_app.py` import this rather than duplicating preprocessing logic, since (unlike the web version) there's no cross-language reason to.
- `digit_image_to_vector()` expects a grayscale (`"L"` mode) `PIL.Image` with a **white background and dark ink** — that's the format both `draw_app.py`'s canvas and `predict.py`'s photo-of-paper images are converted to before calling it.
- It deliberately does not use `Image.crop()`'s built-in out-of-bounds handling (which pads with black) — a manual white-filled paste is used instead, since black padding would otherwise be misread as ink near the edges when a stroke is close to the canvas border.
- `draw_app.py` keeps two parallel representations of the drawing in sync: the visible `tk.Canvas` (for display) and an offscreen `PIL.Image` (`self.image`, built with `ImageDraw`) that mirrors every stroke. This is necessary because a `Canvas` only stores vector drawing commands, not a readable pixel buffer — the `PIL.Image` is what actually gets passed to `digit_image_to_vector()`. Prediction re-runs on every `<B1-Motion>` event (not just on release), so the result panel updates live while drawing.
