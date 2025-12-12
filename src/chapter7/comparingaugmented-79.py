import tensorflow as tf
from keras import models, layers
from keras.datasets import cifar10
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Load and normalize the CIFAR-10 dataset
# ---------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ---------------------------------------------------------------------
# 2. Create tf.data datasets
# ---------------------------------------------------------------------
BATCH_SIZE = 64
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------------------
# 3. Define data augmentation (replaces ImageDataGenerator)
# ---------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1)
])

# ---------------------------------------------------------------------
# 4. Build CNN model
# ---------------------------------------------------------------------
def build_model(use_augmentation=False):
    model = models.Sequential()
    if use_augmentation:
        model.add(data_augmentation)

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------------------------------------------------------------
# 5. Train model without augmentation
# ---------------------------------------------------------------------
model_plain = build_model(use_augmentation=False)
history_plain = model_plain.fit(
    train_ds,
    validation_data=test_ds,
    epochs=20
)

# ---------------------------------------------------------------------
# 6. Train model with augmentation
# ---------------------------------------------------------------------
model_aug = build_model(use_augmentation=True)
history_aug = model_aug.fit(
    train_ds,
    validation_data=test_ds,
    epochs=20
)

model_aug.save("image_classifier.h5")

# ---------------------------------------------------------------------
# 7. Plot comparison
# ---------------------------------------------------------------------
plt.plot(history_plain.history['val_accuracy'], label='Without Augmentation')
plt.plot(history_aug.history['val_accuracy'], label='With Augmentation')
plt.title('Effect of Data Augmentation')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
