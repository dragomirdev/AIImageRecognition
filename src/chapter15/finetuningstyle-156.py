import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ------------------------------------------------------------
# 1. Helper to load and normalize images as float32
# ------------------------------------------------------------
def load_image(path, size=(512, 512)):
    img = Image.open(path).convert('RGB').resize(size)
    img = np.array(img).astype(np.float32) / 255.0  # ✅ ensure float32
    return img[np.newaxis, ...]

content_image = load_image("content.jpg")
style_image = load_image("style.png")

# ------------------------------------------------------------
# 2. Load TF Hub model
# ------------------------------------------------------------
hub_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# ------------------------------------------------------------
# 3. Apply style transfer (must be float32)
# ------------------------------------------------------------
# stylized = hub_model(tf.constant(content_image, dtype=tf.float32),
#                      tf.constant(style_image, dtype=tf.float32))[0]
stylized = hub_model(tf.constant(content_image), tf.constant(style_image) * 0.4)[0]

# ------------------------------------------------------------
# 4. Display results
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(content_image[0])
plt.title("Content Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(style_image[0])
plt.title("Style Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(stylized[0])
plt.title("Stylized Output")
plt.axis('off')

plt.tight_layout()
plt.show()

