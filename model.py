"""CNN definition and image preprocessing shared by training and the web app."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                      # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                      # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def preprocess(img: Image.Image) -> torch.Tensor | None:
    """Turn a drawing (dark strokes on white) into a 1x1x28x28 MNIST-style tensor.

    Mirrors how MNIST was made: crop to the digit, fit it in a 20x20 box
    keeping aspect ratio, then center it by center of mass in a 28x28 frame.
    Returns None if the canvas is blank.
    """
    gray = img.convert("L")
    arr = 255 - np.asarray(gray, dtype=np.float32)  # white ink on black
    arr[arr < 20] = 0  # drop faint noise
    if arr.max() == 0:
        return None

    rows = np.where(arr.max(axis=1) > 0)[0]
    cols = np.where(arr.max(axis=0) > 0)[0]
    arr = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray(arr.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)

    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.asarray(small, dtype=np.float32)

    # shift so the center of mass sits at (14, 14)
    total = canvas.sum()
    ys, xs = np.indices(canvas.shape)
    cy, cx = (canvas * ys).sum() / total, (canvas * xs).sum() / total
    shift_y, shift_x = int(round(14 - cy)), int(round(14 - cx))
    canvas = np.roll(canvas, (shift_y, shift_x), axis=(0, 1))

    canvas /= 255.0
    canvas = (canvas - 0.1307) / 0.3081  # MNIST normalization
    return torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0)
