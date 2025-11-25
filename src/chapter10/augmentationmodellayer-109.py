import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Load CIFAR-10 dataset
# -------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# -------------------------------------------------------------------
# 2. Define augmentation pipeline (applied each epoch dynamically)
# -------------------------------------------------------------------
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2)
])

# -------------------------------------------------------------------
# 3. Build CNN model with augmentation layer at the top
# -------------------------------------------------------------------
model = keras.Sequential([
    data_augmentation,  # 👈 applies random transforms each epoch
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# -------------------------------------------------------------------
# 4. Compile model
# -------------------------------------------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------------------------------------------------
# 5. Train model — augmentation runs automatically each epoch
# -------------------------------------------------------------------
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=64,
                    validation_data=(x_test, y_test))

# -------------------------------------------------------------------
# 6. Visualize sample augmentations (optional)
# -------------------------------------------------------------------
x_batch = x_train[:9]
augmented_batch = data_augmentation(x_batch)
plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(augmented_batch[i])
    plt.axis('off')
plt.suptitle("Example Augmentations Applied Each Epoch")
plt.show()
