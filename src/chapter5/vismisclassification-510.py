import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

# Normalize values (0–255 → 0–1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Show sample images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

x_train_new, x_val, y_train_new, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

history = model.fit(x_train_new, y_train_new,
                    epochs=15,
                    validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.3f}")

predictions = model.predict(x_test)

pred_labels = np.argmax(predictions, axis=1)
incorrect = np.where(pred_labels != y_test)[0]

plt.figure(figsize=(12,6))
for i, idx in enumerate(incorrect[:10]):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[pred_labels[idx]]}")
    plt.axis('off')
plt.show()