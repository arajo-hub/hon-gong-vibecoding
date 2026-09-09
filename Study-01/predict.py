"""
Command-line handwritten digit recognizer.

Loads the weights trained by train_mnist.py and predicts the digit (0-9)
drawn in an input image file.

Usage:
    python predict.py path/to/digit.png
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

WEIGHTS_PATH = Path("weights.json")


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


def predict(weights, x):
    """x: shape (784,) float32 array with pixel values scaled to [0, 1]."""
    z1 = x @ weights["w1"] + weights["b1"]
    a1 = relu(z1)
    z2 = a1 @ weights["w2"] + weights["b2"]
    a2 = softmax(z2)
    return a2


def preprocess_image(image_path: str) -> np.ndarray:
    """Load an image and convert it to a 784-length MNIST-style vector.

    Expects a digit drawn on a light background (like a photo of handwriting
    on paper). The image is converted to grayscale, resized to 28x28, and
    inverted so the digit is white (high value) on a black background,
    matching the MNIST training data format.
    """
    img = Image.open(image_path).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32) / 255.0
    return pixels.reshape(-1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    weights = load_weights()
    x = preprocess_image(image_path)
    probs = predict(weights, x)

    predicted_digit = int(np.argmax(probs))
    confidence = float(probs[predicted_digit]) * 100

    print(f"Predicted digit: {predicted_digit} (confidence: {confidence:.1f}%)")
    print("Class probabilities:")
    for digit, p in enumerate(probs):
        print(f"  {digit}: {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
