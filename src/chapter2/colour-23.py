from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load image in RGB mode
image = Image.open('cat.jpg').convert('RGB')

# Convert to NumPy array
image_array = np.array(image)

# Split into R, G, B channels
r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

# Display all channels
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(image_array)
plt.title("Original")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(r, cmap='Reds')
plt.title("Red Channel")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(g, cmap='Greens')
plt.title("Green Channel")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(b, cmap='Blues')
plt.title("Blue Channel")
plt.axis('off')

plt.tight_layout()
plt.show()
