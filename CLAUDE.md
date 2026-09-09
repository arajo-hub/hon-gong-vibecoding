# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

A collection of small, independent coding studies, each in its own numbered `Study-NN/` directory. `Study-01` is the only study so far: a handwritten digit recognizer built from scratch with NumPy (no ML framework).

## Commands (Study-01)

Study-01 uses a local virtualenv at `Study-01/.venv` (gitignored, not committed).

```bash
# One-time setup
python3 -m venv Study-01/.venv
Study-01/.venv/bin/pip install numpy pillow

# Train the model — downloads MNIST (mnist.npz, gitignored) on first run,
# then overwrites Study-01/weights.json with freshly trained weights
cd Study-01 && ../Study-01/.venv/bin/python train_mnist.py
```

`web_version/` and `desktop_version/` each have their own `CLAUDE.md` with the commands to run that version — see those before working inside either directory. There is no test suite, linter, or build step in this repository.

## Architecture (Study-01)

- `train_mnist.py` — trains a 784 → 128 (ReLU) → 10 (softmax) MLP with a hand-written forward/backward pass and mini-batch SGD + momentum (no TensorFlow/PyTorch). Downloads MNIST from a fixed Google-hosted URL, then exports the trained weights to `weights.json`, rounded to 4 decimals to keep the file small.
- `weights.json` — the single source of truth for trained weights, shared by every consumer: `desktop_version/model.py` and the `web_version/app.js` browser demo (which fetches it at runtime rather than embedding it, and requires a local HTTP server — see `web_version/CLAUDE.md`). Retraining overwrites this one file; nothing else needs to change. `w1` is shaped `(784, 128)` indexed `[input][hidden]`; `w2` is shaped `(128, 10)` indexed `[hidden][output]`. Any reimplementation of the forward pass (Python, JS, or otherwise) must stay numerically consistent with `MLP.forward` in `train_mnist.py`.
- `web_version/` — browser UI with a drawing canvas; runs the forward pass client-side in vanilla JS. Also published as a standalone Claude Artifact (see the link in `README.md`) that embeds its own copy of `weights.json` inline instead of fetching it.
- `desktop_version/` — native Tkinter UI with a drawing canvas (`draw_app.py`) plus a CLI for classifying image files (`predict.py`); both share preprocessing/inference code via `model.py`.
