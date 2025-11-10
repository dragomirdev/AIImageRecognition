from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load image and convert to grayscale
image = Image.open('cat.jpg').convert('L')  # 'L' mode = 8-bit grayscale

# Convert to NumPy array for numerical access if needed
image_array = np.array(image)

print("Image shape:", image_array.shape)

# Display image
plt.imshow(image_array, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()