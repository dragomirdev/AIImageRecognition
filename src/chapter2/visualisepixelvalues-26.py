from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image_gray = Image.open('cat.jpg').convert('L')  # 'L' = 8-bit grayscale

# Convert to NumPy array
image_array = np.array(image_gray)

# Display as a heatmap
plt.imshow(image_array, cmap='hot')
plt.title("Pixel Intensity Heatmap")
plt.axis('off')
plt.show()