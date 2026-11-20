"""Flask app: serves the drawing page and a /predict endpoint."""
import base64
import io

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

from model import DigitCNN, preprocess

app = Flask(__name__)

# Below this top-class probability we ask the user to redraw instead of guessing.
MIN_CONFIDENCE = 0.50

model = DigitCNN()
model.load_state_dict(torch.load("digit_cnn.pt", map_location="cpu"))
model.eval()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    data = (request.get_json(silent=True) or {}).get("image", "")
    try:
        raw = base64.b64decode(data.split(",", 1)[-1], validate=True)
        img = Image.open(io.BytesIO(raw))
        # flatten any transparency onto white
        bg = Image.new("RGBA", img.size, "white")
        img = Image.alpha_composite(bg, img.convert("RGBA"))
    except Exception:
        return jsonify(error="Could not read the image."), 400

    x = preprocess(img)
    if x is None:
        return jsonify(error="Draw a digit first!"), 400

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    digit = int(probs.argmax())
    confidence = float(probs[digit])
    return jsonify(
        digit=digit,
        confidence=confidence,
        uncertain=confidence < MIN_CONFIDENCE,
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
