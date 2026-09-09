"""
Shared inference code for the desktop apps (predict.py, draw_app.py).

Loads the weights trained by ../train_mnist.py and runs the same
784 -> 128 (ReLU) -> 10 (softmax) forward pass, plus an MNIST-style
preprocessing step (crop to the ink bounding box, then resize to 28x28)
shared by both the CLI and the drawing GUI.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights.json"


def load_weights():
    with open(WEIGHTS_PATH) as f:
        weights = json.load(f)
    return {
        "w1": np.array(weights["w1"], dtype=np.float32),
        "b1": np.array(weights["b1"], dtype=np.float32),
        "w2": np.array(weights["w2"], dtype=np.float32),
        "b2": np.array(weights["b2"], dtype=np.float32),
    }


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def forward(weights, x):
    """x: shape (784,) float32 array, MNIST-style (digit bright on a black background)."""
    z1 = x @ weights["w1"] + weights["b1"]
    a1 = relu(z1)
    z2 = a1 @ weights["w2"] + weights["b2"]
    return softmax(z2)


def digit_image_to_vector(ink_image: Image.Image) -> np.ndarray | None:
    """Convert a hand-drawn digit into a 784-length MNIST-style vector.

    `ink_image` must be a grayscale ("L") image with a white (255) background
    and dark ink strokes, matching what draw_app.py paints. The ink's
    bounding box is cropped with padding and resized to 28x28, then
    inverted so the digit is bright on a black background (MNIST's format).
    Returns None if the image is blank (nothing drawn).
    """
    pixels = np.array(ink_image, dtype=np.float32)
    ink_mask = pixels < 250  # anything noticeably darker than the white background
    ys, xs = np.nonzero(ink_mask)
    if len(xs) == 0:
        return None

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    box_w, box_h = max_x - min_x + 1, max_y - min_y + 1
    size = max(box_w, box_h)
    pad = size * 0.28
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    half = size / 2 + pad

    # Crop manually onto a white canvas: Image.crop() fills out-of-bounds
    # regions with black, which would be misread as ink at the edges.
    side = round(half * 2)
    left, top = round(cx - half), round(cy - half)
    square = Image.new("L", (side, side), color=255)
    src_box = (
        max(left, 0), max(top, 0),
        min(left + side, ink_image.width), min(top + side, ink_image.height),
    )
    if src_box[2] > src_box[0] and src_box[3] > src_box[1]:
        region = ink_image.crop(src_box)
        square.paste(region, (src_box[0] - left, src_box[1] - top))

    resized = square.resize((28, 28), Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32) / 255.0
    return (1.0 - arr).reshape(-1)  # invert: dark ink -> bright value
