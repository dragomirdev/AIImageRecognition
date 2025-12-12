import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("🐾 Streamlit Image Classifier Demo")

# Load your CIFAR-10 model
model = tf.keras.models.load_model("image_classifier.h5")

# Correct CIFAR-10 class names
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

uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    # Convert to RGB and resize to CIFAR-10 dimensions
    img = Image.open(uploaded).convert("RGB").resize((32, 32))

    # Show the resized image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess input
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    preds = model.predict(img_array)
    label = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    st.success(f"Prediction: **{label}** ({confidence * 100:.2f}% confidence)")
