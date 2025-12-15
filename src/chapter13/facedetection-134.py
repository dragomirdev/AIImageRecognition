import mediapipe as mp
from PIL import Image, ImageDraw
import numpy as np

# Load image
img = Image.open("people.jpg")
img_rgb = np.array(img)

# Mediapipe face detection
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

results = detector.process(img_rgb)

# Draw bounding boxes
draw = ImageDraw.Draw(img)

if results.detections:
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = img_rgb.shape

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)

        draw.rectangle((x, y, x + w_box, y + h_box), outline="red", width=3)

img.show()
