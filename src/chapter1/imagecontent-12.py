from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load an image using Pillow
image = Image.open('cat.jpg')

# Convert to NumPy array (for shape info and matplotlib compatibility)
image_array = np.array(image)

print("Image shape:", image_array.shape)

# Display it
plt.imshow(image_array)
plt.title("A Cat as Numbers")
plt.axis('off')
plt.show()