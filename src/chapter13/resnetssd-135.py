import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

# Load SSD300 model locally (NO INTERNET)
model = ssd300_vgg16(weights="SSD300_VGG16_Weights.DEFAULT")
model.eval()

# COCO labels
labels = [
    "__background__",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair dryer", "toothbrush"
]

# Load image
img = Image.open("people.jpg").convert("RGB")

# Preprocess
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])
input_tensor = transform(img).unsqueeze(0)

# Run inference
with torch.no_grad():
    outputs = model(input_tensor)[0]

draw = ImageDraw.Draw(img)

# Extract boxes, scores, labels
boxes = outputs["boxes"].numpy()
scores = outputs["scores"].numpy()
classes = outputs["labels"].numpy()

CONF_THRESH = 0.5

for box, score, cls in zip(boxes, scores, classes):
    if score < CONF_THRESH:
        continue

    # Filter for "person" since SSD COCO does not include "face"
    if labels[cls] != "person":
        continue

    x1, y1, x2, y2 = box.astype(int)
    draw.rectangle((x1, y1, x2, y2), outline="green", width=3)
    draw.text((x1, y1), f"{labels[cls]} {score:.2f}", fill="red")


img.show()


