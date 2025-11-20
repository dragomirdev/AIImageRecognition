from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from keras import layers, models
from keras.layers import BatchNormalization
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# 1. Load CIFAR-10 dataset
# ---------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values (0-255 → 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ---------------------------------------------------------------------
# 2. Create TensorFlow datasets
# ---------------------------------------------------------------------
BATCH_SIZE = 64
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------------------
# 3. Data augmentation using modern Keras layers
# ---------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

# ---------------------------------------------------------------------
# 4. Build the CNN model
# ---------------------------------------------------------------------
model = models.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------------------------------------------------
# 5. Train the model
# ---------------------------------------------------------------------
history = model.fit(train_ds,
                    epochs=20,
                    validation_data=test_ds)

# ---------------------------------------------------------------------
# 6. Evaluate and confusion matrix
# ---------------------------------------------------------------------
y_pred = model.predict(x_test, batch_size=BATCH_SIZE)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
