# VibeCoding

Small coding studies and experiments.

## Study-01 — Handwritten digit recognizer

An MNIST digit recognizer built from scratch with NumPy (no ML framework), with separate web and desktop apps:

- `Study-01/train_mnist.py` — trains a 784→128 (ReLU)→10 (softmax) MLP on MNIST, ~98% test accuracy.
- `Study-01/weights.json` — the trained weights, shared by both apps below.
- `Study-01/web_version/` — draw a digit in the browser and see it recognized live (vanilla JS, no server-side inference). See its `CLAUDE.md` for how to serve it locally.
- `Study-01/desktop_version/` — a native Tkinter drawing app (`draw_app.py`), plus a CLI (`predict.py`) that classifies an existing image file.

Live demo (same as `web_version/`, published as a Claude Artifact): https://claude.ai/code/artifact/ae262491-4b7f-419d-a082-aad1360d39f9
