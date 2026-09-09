"""
Desktop handwritten digit recognizer with a drawing canvas.

Draw a digit with the mouse; the trained network (loaded from
../weights.json) predicts it live as you draw.
"""

import tkinter as tk
from tkinter import font as tkfont

from PIL import Image, ImageDraw

from model import digit_image_to_vector, forward, load_weights

CANVAS_SIZE = 280
BRUSH_WIDTH = 18


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ink Reader — Desktop")
        self.weights = load_weights()

        mono = tkfont.Font(family="Menlo", size=13)
        mono_big = tkfont.Font(family="Menlo", size=48, weight="bold")
        label_font = tkfont.Font(family="Helvetica", size=11)

        # Parallel offscreen image: strokes are mirrored onto this bitmap
        # so we have real pixel data to run inference on (a Tk Canvas holds
        # only vector drawing commands, not a readable pixel buffer).
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)

        container = tk.Frame(root, padx=16, pady=16)
        container.pack()

        left = tk.Frame(container)
        left.grid(row=0, column=0, padx=(0, 16))

        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                 bg="white", cursor="crosshair",
                                 highlightthickness=1, highlightbackground="#999")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        controls = tk.Frame(left)
        controls.pack(fill="x", pady=(8, 0))
        tk.Label(controls, text="Draw a digit, 0-9", font=label_font, fg="#666").pack(side="left")
        tk.Button(controls, text="Clear", command=self.clear).pack(side="right")

        right = tk.Frame(container)
        right.grid(row=0, column=1, sticky="n")

        tk.Label(right, text="RECOGNIZED AS", font=label_font, fg="#666").pack(anchor="w")
        self.result_label = tk.Label(right, text="–", font=mono_big, fg="#33409e")
        self.result_label.pack(anchor="w", pady=(0, 4))
        self.confidence_label = tk.Label(right, text="confidence –", font=mono, fg="#666")
        self.confidence_label.pack(anchor="w", pady=(0, 12))

        self.bar_rows = []
        bars_frame = tk.Frame(right)
        bars_frame.pack(anchor="w")
        for digit in range(10):
            row = tk.Frame(bars_frame)
            row.pack(fill="x", pady=1)
            digit_label = tk.Label(row, text=str(digit), font=mono, width=2, anchor="w")
            digit_label.pack(side="left")
            track = tk.Canvas(row, width=160, height=10, bg="#eee", highlightthickness=0)
            track.pack(side="left", padx=6)
            bar = track.create_rectangle(0, 0, 0, 10, fill="#33409e", width=0)
            pct_label = tk.Label(row, text="0%", font=mono, width=5, anchor="e")
            pct_label.pack(side="left")
            self.bar_rows.append((digit_label, track, bar, pct_label))

        self.last_xy = None

    def on_press(self, event):
        self.last_xy = (event.x, event.y)

    def on_drag(self, event):
        x, y = event.x, event.y
        if self.last_xy is not None:
            x0, y0 = self.last_xy
            self.canvas.create_line(x0, y0, x, y, width=BRUSH_WIDTH,
                                     fill="black", capstyle=tk.ROUND, joinstyle=tk.ROUND)
            self.draw.line([x0, y0, x, y], fill=0, width=BRUSH_WIDTH)
            r = BRUSH_WIDTH / 2
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
        self.last_xy = (x, y)
        self.predict()

    def on_release(self, event):
        self.last_xy = None
        self.predict()

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)
        self.result_label.config(text="–")
        self.confidence_label.config(text="confidence –")
        for _, track, bar, pct_label in self.bar_rows:
            track.coords(bar, 0, 0, 0, 10)
            pct_label.config(text="0%")

    def predict(self):
        vec = digit_image_to_vector(self.image)
        if vec is None:
            return
        probs = forward(self.weights, vec)
        best = int(probs.argmax())

        self.result_label.config(text=str(best))
        self.confidence_label.config(text=f"confidence {probs[best] * 100:.1f}%")

        for digit, (digit_label, track, bar, pct_label) in enumerate(self.bar_rows):
            pct = float(probs[digit]) * 100
            width = int(160 * pct / 100)
            track.coords(bar, 0, 0, width, 10)
            pct_label.config(text=f"{pct:.0f}%")
            digit_label.config(fg="#33409e" if digit == best else "black")


def main():
    root = tk.Tk()
    root.resizable(False, False)
    DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
