from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the image (Pillow uses RGB by default)
image = Image.open('cat.jpg').convert('RGB')

# Rotate 90 degrees clockwise
rotated = image.rotate(-90, expand=True)  # Negative = clockwise

# Convert to NumPy array for matplotlib display
rotated_array = np.array(rotated)

# Display
plt.imshow(rotated_array)
plt.title("Rotated 90°")
plt.axis('off')
plt.show()