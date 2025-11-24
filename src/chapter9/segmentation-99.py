import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------
# 1. Load pre-trained Mask R-CNN (Inception-ResNet backbone)
# ---------------------------------------------------------------------
MODEL_URL = "https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1"
model = hub.load(MODEL_URL)
print("✅ Model loaded successfully")

# ---------------------------------------------------------------------
# 2. Load and prepare image
# ---------------------------------------------------------------------
IMAGE_PATH = "people.jpeg"
image = Image.open(IMAGE_PATH).convert("RGB")
img = np.array(image)

# Convert to tensor (dtype=uint8 and add batch dimension)
input_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)[tf.newaxis, ...]

# ---------------------------------------------------------------------
# 3. Run model
# ---------------------------------------------------------------------
outputs = model(input_tensor)
result = {key: value.numpy() for key, value in outputs.items()}

# Extract detections
boxes = result["detection_boxes"][0]
masks = result["detection_masks"][0]
scores = result["detection_scores"][0]
classes = result["detection_classes"][0]

# ---------------------------------------------------------------------
# 4. Visualize segmentation masks
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.title("Mask R-CNN Segmentation")
plt.axis('off')

# Overlay masks for top confident detections
for i in range(min(5, len(scores))):
    if scores[i] < 0.5:
        continue
    mask = masks[i]
    plt.imshow(mask, alpha=0.5, cmap='cool')

plt.show()


