from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Load image using Pillow
image = Image.open("noisy_image.jpeg").convert("RGB")
img = np.array(image)

# Apply Gaussian blur (denoising) using SciPy
denoised = gaussian_filter(img, sigma=(1, 1, 0))  # sigma≈blur strength

# Display images
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(denoised.astype(np.uint8))
plt.title("Denoised")
plt.axis('off')

plt.show()
