import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import numpy as np

# -------------------------------------------------------------------
# 1. Load CIFAR-10 dataset
# -------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Convert to float32 (NumPy 2.2 safe) and normalize
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

# Ensure labels are int64 for compatibility
y_train = y_train.astype(np.int64)
y_test  = y_test.astype(np.int64)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# -------------------------------------------------------------------
# 2. Data augmentation layer
# -------------------------------------------------------------------
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# -------------------------------------------------------------------
# 3. Build CNN model with augmentation
# -------------------------------------------------------------------
model = models.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# -------------------------------------------------------------------
# 4. Compile the model
# -------------------------------------------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------------------------------------------
# 5. Train the model
# -------------------------------------------------------------------
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# -------------------------------------------------------------------
# 6. Evaluate with classification report
# -------------------------------------------------------------------
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

print("\n✅ Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))
