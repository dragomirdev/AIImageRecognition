from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("image_classifier.h5")
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # Load image, remove alpha, resize to CIFAR-10 size
    img = Image.open(file).convert("RGB").resize((32, 32))

    # Convert to array, normalize, add batch dimension
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    preds = model.predict(img_array)

    # Extract label + confidence
    label = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    return jsonify({
        'prediction': label,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
