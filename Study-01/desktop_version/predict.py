"""
Command-line handwritten digit recognizer.

Loads the weights trained by ../train_mnist.py and predicts the digit
(0-9) drawn in an input image file.

Usage:
    python predict.py path/to/digit.png
"""

import sys

import numpy as np
from PIL import Image

from model import digit_image_to_vector, forward, load_weights


def preprocess_image(image_path: str) -> np.ndarray | None:
    """Load an image (e.g. a photo of handwriting on paper) and convert it
    to a 784-length MNIST-style vector, or None if it looks blank."""
    img = Image.open(image_path).convert("L")
    # Photos of pen-on-paper are usually a dark digit on a light background,
    # which is exactly what digit_image_to_vector() expects.
    return digit_image_to_vector(img)


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    weights = load_weights()
    x = preprocess_image(image_path)
    if x is None:
        print("No digit detected in the image (it looks blank).")
        sys.exit(1)

    probs = forward(weights, x)
    predicted_digit = int(np.argmax(probs))
    confidence = float(probs[predicted_digit]) * 100

    print(f"Predicted digit: {predicted_digit} (confidence: {confidence:.1f}%)")
    print("Class probabilities:")
    for digit, p in enumerate(probs):
        print(f"  {digit}: {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
