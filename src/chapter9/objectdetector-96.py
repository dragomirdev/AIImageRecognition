import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image

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

# ---------------------------------------------------------------------
# 5. Display detections (top 5)
# ---------------------------------------------------------------------
print(f"\n--- Detections for {IMAGE_PATH} ---")
for i in range(15):
    class_id = int(class_ids[i])
    label = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else f"ID {class_id}"
    print(f"{i+1}. {label}: {scores[i]:.2f}")

