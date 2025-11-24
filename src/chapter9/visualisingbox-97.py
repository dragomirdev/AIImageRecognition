import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Load pre-trained SSD MobileNet V2 model from TensorFlow Hub
# ---------------------------------------------------------------------
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
detector = hub.load(MODEL_URL)
print("✅ Model loaded successfully")

# ---------------------------------------------------------------------
# 2. Load image (ensure RGB)
# ---------------------------------------------------------------------
IMAGE_PATH = "dog.png"  # change to your file path
image = Image.open(IMAGE_PATH).convert("RGB")
img = np.array(image)

# Model expects dtype=uint8 with shape (1, h, w, 3)
input_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)[tf.newaxis, ...]

# ---------------------------------------------------------------------
# 3. Run detection
# ---------------------------------------------------------------------
outputs = detector(input_tensor)
result = {k: v.numpy() for k, v in outputs.items()}

boxes = result["detection_boxes"][0]
class_ids = result["detection_classes"][0]
scores = result["detection_scores"][0]

# ---------------------------------------------------------------------
# 4. COCO labels for SSD MobileNet
# ---------------------------------------------------------------------
COCO_LABELS = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Define class_names from detected IDs
class_names = [
    COCO_LABELS[int(cls)] if int(cls) < len(COCO_LABELS) else f"ID {int(cls)}"
    for cls in class_ids
]

# ---------------------------------------------------------------------
# 5. Display detections (top 15)
# ---------------------------------------------------------------------
print(f"\n--- Detections for {IMAGE_PATH} ---")
for i in range(15):
    print(f"{i+1}. {class_names[i]}: {scores[i]:.2f}")

# ---------------------------------------------------------------------
# 6. Visualize bounding boxes and class names
# ---------------------------------------------------------------------
fig, ax = plt.subplots(1)
ax.imshow(img)

for i in range(3):  # top 3 boxes
    if scores[i] < 0.5:
        continue
    y_min, x_min, y_max, x_max = boxes[i]
    (left, right, top, bottom) = (x_min * img.shape[1], x_max * img.shape[1],
                                  y_min * img.shape[0], y_max * img.shape[0])
    rect = patches.Rectangle((left, top), right-left, bottom-top,
                             linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.text(left, top - 10, f"{class_names[i]} {scores[i]:.2f}",
            color='black', fontsize=10, weight='bold', backgroundcolor='lime')

plt.axis('off')
plt.title(f"SSD MobileNet V2 Detections for {IMAGE_PATH}")
plt.show()
