"""
Train a small MLP (Multi-Layer Perceptron) on the MNIST handwritten digit
dataset using only NumPy, then export the learned weights to a compact
JSON file so they can be embedded in a browser-based demo.

Network architecture:
    input (784) -> hidden (128, ReLU) -> output (10, softmax)
"""

import json
import urllib.request
from pathlib import Path

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_PATH = Path("mnist.npz")
WEIGHTS_PATH = Path("weights.json")

HIDDEN_SIZE = 128
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
SEED = 42


def download_mnist() -> None:
    if DATA_PATH.exists():
        return
    print(f"Downloading MNIST dataset from {MNIST_URL} ...")
    urllib.request.urlretrieve(MNIST_URL, DATA_PATH)
    print("Download complete.")


def load_mnist():
    with np.load(DATA_PATH) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    # Flatten 28x28 images to 784-length vectors and scale pixels to [0, 1].
    x_train = x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0
    x_test = x_test.reshape(len(x_test), -1).astype(np.float32) / 255.0

    # One-hot encode the labels.
    y_train_oh = np.eye(10, dtype=np.float32)[y_train]

    return x_train, y_train_oh, y_test, x_test


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(np.float32)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class MLP:
    """A minimal two-layer neural network trained with mini-batch SGD."""

    def __init__(self, input_size, hidden_size, output_size, rng):
        # He initialization for ReLU layers.
        self.w1 = rng.standard_normal((input_size, hidden_size)).astype(np.float32) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = rng.standard_normal((hidden_size, output_size)).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size, dtype=np.float32)

        # Momentum buffers.
        self.vw1 = np.zeros_like(self.w1)
        self.vb1 = np.zeros_like(self.b1)
        self.vw2 = np.zeros_like(self.w2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, x):
        z1 = x @ self.w1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = softmax(z2)
        return z1, a1, a2

    def predict(self, x):
        _, _, a2 = self.forward(x)
        return np.argmax(a2, axis=1)

    def train_step(self, x, y_onehot, lr, momentum):
        batch_size = x.shape[0]

        z1, a1, a2 = self.forward(x)

        # Cross-entropy loss gradient w.r.t. softmax input simplifies to (a2 - y).
        d_z2 = (a2 - y_onehot) / batch_size
        d_w2 = a1.T @ d_z2
        d_b2 = d_z2.sum(axis=0)

        d_a1 = d_z2 @ self.w2.T
        d_z1 = d_a1 * relu_derivative(z1)
        d_w1 = x.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        # SGD with momentum.
        self.vw1 = momentum * self.vw1 - lr * d_w1
        self.vb1 = momentum * self.vb1 - lr * d_b1
        self.vw2 = momentum * self.vw2 - lr * d_w2
        self.vb2 = momentum * self.vb2 - lr * d_b2

        self.w1 += self.vw1
        self.b1 += self.vb1
        self.w2 += self.vw2
        self.b2 += self.vb2

        loss = -np.sum(y_onehot * np.log(a2 + 1e-9)) / batch_size
        return loss


def train():
    rng = np.random.default_rng(SEED)

    download_mnist()
    x_train, y_train_oh, y_test, x_test = load_mnist()

    model = MLP(input_size=784, hidden_size=HIDDEN_SIZE, output_size=10, rng=rng)

    num_samples = x_train.shape[0]
    steps_per_epoch = num_samples // BATCH_SIZE

    for epoch in range(1, EPOCHS + 1):
        perm = rng.permutation(num_samples)
        x_shuffled = x_train[perm]
        y_shuffled = y_train_oh[perm]

        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            start = step * BATCH_SIZE
            end = start + BATCH_SIZE
            batch_x = x_shuffled[start:end]
            batch_y = y_shuffled[start:end]
            epoch_loss += model.train_step(batch_x, batch_y, LEARNING_RATE, MOMENTUM)

        avg_loss = epoch_loss / steps_per_epoch
        test_preds = model.predict(x_test)
        test_acc = np.mean(test_preds == y_test)
        print(f"Epoch {epoch:2d}/{EPOCHS} - loss: {avg_loss:.4f} - test accuracy: {test_acc * 100:.2f}%")

    export_weights(model)
    return model


def export_weights(model: MLP, decimals: int = 4) -> None:
    """Save rounded weights to JSON so the file stays compact for embedding in a web page."""
    weights = {
        "w1": np.round(model.w1, decimals).tolist(),
        "b1": np.round(model.b1, decimals).tolist(),
        "w2": np.round(model.w2, decimals).tolist(),
        "b2": np.round(model.b2, decimals).tolist(),
        "hidden_size": HIDDEN_SIZE,
    }
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights, f)
    size_kb = WEIGHTS_PATH.stat().st_size / 1024
    print(f"Saved trained weights to {WEIGHTS_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    train()
