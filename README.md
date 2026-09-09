# VibeCoding

Small coding studies and experiments.

## Study-01 — Handwritten digit recognizer

An MNIST digit recognizer built from scratch with NumPy (no ML framework):

- `Study-01/train_mnist.py` — trains a 784→128 (ReLU)→10 (softmax) MLP on MNIST, ~98% test accuracy.
- `Study-01/predict.py` — CLI that predicts the digit in an image file using the trained weights.
- `Study-01/weights.json` — the trained weights (also embedded in the live demo below, running entirely in the browser).

Live demo (draw a digit and see it recognized in real time): https://claude.ai/code/artifact/ae262491-4b7f-419d-a082-aad1360d39f9
