import numpy as np
from PIL import Image

# Load image in RGB mode
image = Image.open('cat.jpg').convert('RGB')

# Convert to NumPy array
image_array = np.array(image)

print(type(image_array))
print("Image shape:", image_array.shape)
print("Pixel [0, 0]:", image_array[0, 0])