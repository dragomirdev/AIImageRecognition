
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Example: preprocess CIFAR-10 images
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Define modern augmentation pipeline
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
    layers.RandomFlip("horizontal"),
    layers.RandomContrast(0.1),
])

# Pick a small batch of training images
x_batch = x_train[:90]

# Apply augmentation
augmented_batch = data_augmentation(x_batch)

# Visualize 9 examples
plt.figure(figsize=(6,6))
for i in range(90):
    plt.subplot(10,9,i+1)
    plt.imshow(augmented_batch[i])
    plt.axis('off')
plt.suptitle("Augmented Images (Modern Keras API)")
plt.show()

