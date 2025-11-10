import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def adjust_brightness(image, beta):
    """Increase or decrease brightness by adding beta."""
    result = np.clip(image.astype(np.float32) + beta, 0, 255)
    return result.astype(np.uint8)

def adjust_contrast(image, alpha):
    """Adjust contrast by scaling pixel values."""
    result = np.clip(image.astype(np.float32) * alpha, 0, 255)
    return result.astype(np.uint8)

# Load image in RGB mode
image = Image.open('cat.jpg').convert('RGB')

# Convert to NumPy array
image_rgb = np.array(image)

# Apply adjustments
bright = adjust_brightness(image_rgb, beta=50)
dark = adjust_brightness(image_rgb, beta=-50)
high_contrast = adjust_contrast(image_rgb, alpha=1.5)

# Display
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(bright); plt.title("Brighter"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(dark); plt.title("Darker"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(high_contrast); plt.title("Higher Contrast"); plt.axis('off')
plt.show()