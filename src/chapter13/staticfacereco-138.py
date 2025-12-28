from mtcnn import MTCNN
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# 1. Load image
# -------------------------------------------------------------
image_path = "people.jpg"  # 👈 replace with your file
image = Image.open(image_path).convert("RGB")
img_array = np.array(image)

# -------------------------------------------------------------
# 2. Initialize MTCNN detector
# -------------------------------------------------------------
detector = MTCNN()
results = detector.detect_faces(img_array)

# -------------------------------------------------------------
# 3. Draw detected face boxes
# -------------------------------------------------------------
draw = ImageDraw.Draw(image)
for res in results:
    x, y, w, h = res["box"]
    draw.rectangle([x, y, x+w, y+h], outline="red", width=3)

    # Optional: draw confidence score
    score = res["confidence"]
    draw.text((x, y - 10), f"{score:.2f}", fill="yellow")

# -------------------------------------------------------------
# 4. Display the result
# -------------------------------------------------------------
plt.imshow(image)
plt.axis("off")
plt.title("Face Detection using MTCNN")
plt.show()
