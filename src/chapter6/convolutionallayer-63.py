from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Load grayscale image
image = Image.open('cat.jpg').convert('L')
image_array = np.array(image)

# Define edge detection filter 
kernel = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]])

# Apply 2D convolution
edges = convolve2d(image_array, kernel, mode='same', boundary='symm')

# Normalize result for display (clip to 0–255 range)
edges = np.clip(edges, 0, 255).astype(np.uint8)

# Display original and edge-detected images
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(image_array, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection")
plt.axis('off')

plt.show()
