from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load image (Pillow uses RGB by default)
image = Image.open('cat.jpg').convert('RGB')

# Resize to 200x200 pixels
resized = image.resize((400, 200))

# Convert to NumPy array for Matplotlib display
resized_array = np.array(resized)

# Display
plt.imshow(resized_array)
plt.title("Resized to 400×200")
plt.axis('off')
plt.show()