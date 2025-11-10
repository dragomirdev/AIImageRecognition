from PIL import Image
import numpy as np

# Load the image and convert to RGB
image = Image.open('cat.jpg').convert('RGB')

# Convert to NumPy array
image_array = np.array(image)

# Normalize pixel values (0–255 → 0–1)
image_normalized = image_array / 255.0

# Show example pixel before and after normalization
print("Pixel value before:", image_array[0, 0])          # e.g. [125 132 140]
print("Pixel value after normalization:", image_normalized[0, 0])  # e.g. [0.49 0.52 0.55]
