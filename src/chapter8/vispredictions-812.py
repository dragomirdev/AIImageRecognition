import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16


# ---------------------------------------------------------------------
# 1. Load CIFAR-10 and normalize
# ---------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Class names (for predictions/plotting)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ---------------------------------------------------------------------
# 2. Resize CIFAR-10 images for VGG16 (expects 224×224×3)
# ---------------------------------------------------------------------
x_train_resized = tf.image.resize(x_train, (224, 224))
x_test_resized = tf.image.resize(x_test, (224, 224))

# ---------------------------------------------------------------------
# 3. Load VGG16 base (transfer learning)
# ---------------------------------------------------------------------
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze pretrained weights

# ---------------------------------------------------------------------
# 4. Add custom classification layers
# ---------------------------------------------------------------------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# ---------------------------------------------------------------------
# 5. Compile and train
# ---------------------------------------------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train_resized, y_train,
    epochs=10,
    validation_data=(x_test_resized, y_test)
)

# ---------------------------------------------------------------------
# 6. Make predictions
# ---------------------------------------------------------------------
predictions = model.predict(x_test_resized)

# ---------------------------------------------------------------------
# 7. Plot sample predictions
# ---------------------------------------------------------------------
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    pred = class_names[np.argmax(predictions[i])]
    true = class_names[y_test[i][0]]
    color = 'green' if pred == true else 'red'
    plt.title(f"Pred: {pred}\nTrue: {true}", color=color)
    plt.axis('off')

plt.suptitle("VGG16 Transfer Learning on CIFAR-10", fontsize=16)
plt.tight_layout()
plt.show()
