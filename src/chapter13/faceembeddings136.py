from keras_facenet import FaceNet
from mtcnn import MTCNN
from PIL import Image
import numpy as np

# Load FaceNet embedder
embedder = FaceNet()

# MTCNN detector (pure Python + TensorFlow backend)
detector = MTCNN()

def extract_face(image_path):
    """
    Loads an image with PIL, detects a face with MTCNN,
    and returns a 160x160 normalized NumPy array for FaceNet.
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img)

    results = detector.detect_faces(img_np)

    if len(results) == 0:
        raise ValueError(f"No face found in {image_path}")

    # Get bounding box from first face
    x, y, w, h = results[0]["box"]

    # Crop and resize
    face = img_np[y:y+h, x:x+w]
    face = Image.fromarray(face).resize((160, 160))
    face = np.asarray(face).astype("float32") / 255.0

    return face

def get_embedding(face):
    """
    Converts a cropped face to a 512-dimension embedding.
    """
    return embedder.embeddings([face])[0]

def compare_faces(img1_path, img2_path, threshold=0.9):
    """
    Returns True if the two images show the same person.
    """
    face1 = extract_face(img1_path)
    face2 = extract_face(img2_path)

    emb1 = get_embedding(face1)
    emb2 = get_embedding(face2)

    distance = np.linalg.norm(emb1 - emb2)
    print("Distance:", distance)

    return distance < threshold

# Test
match = compare_faces("elon.jpeg", "unknown.jpeg")

print("Match found!" if match else "No match.")
