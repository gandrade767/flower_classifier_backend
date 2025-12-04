import io
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf
from keras.layers import TFSMLayer
from flask_cors import CORS

IMG_HEIGHT = 180
IMG_WIDTH = 180

app = Flask(__name__)
CORS(app)

print("Carregando modelo SavedModel com TFSMLayer...")
model = TFSMLayer("flower_model", call_endpoint="serve")

print("Carregando classes...")
with open("class_names.json", "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    img_array = preprocess_image(img_bytes)

    # Agora sabemos que output é um tensor
    output = model(img_array)          # shape (1, 5)
    predictions = output[0].numpy()    # vira array shape (5,)

    probabilities = tf.nn.softmax(predictions).numpy()

    idx = int(np.argmax(probabilities))
    classe = CLASS_NAMES[idx]
    confianca = float(probabilities[idx])

    return jsonify({
        "class": classe,
        "confidence": confianca,
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
        }
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)

