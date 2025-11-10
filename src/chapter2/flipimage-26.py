from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the image (Pillow uses RGB by default)
image = Image.open('cat.jpg').convert('RGB')

# Flip TOP BOTTOM
flipped = image.transpose(Image.FLIP_TOP_BOTTOM)

# Convert to NumPy array for matplotlib display
flipped_array = np.array(flipped)

# Display
plt.imshow(flipped_array)
plt.title("Flipped Top-Bottom")
plt.axis('off')
plt.show()