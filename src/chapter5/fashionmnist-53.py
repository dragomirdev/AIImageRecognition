import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

# Normalize values (0–255 → 0–1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Show sample images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.show()

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),          # Convert 2D → 1D
    layers.Dense(256, activation='relu'),          # Hidden layer 1
    layers.Dropout(0.3),                           # Regularization
    layers.Dense(128, activation='relu'),          # Hidden layer 2
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')         # Output layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20,
                    validation_data=(x_test, y_test))

