import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import shap
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Load CIFAR-10
# -------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype(np.float32)/255.0, x_test.astype(np.float32)/255.0
y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# -------------------------------------------------------------------
# 2. Build CNN using Functional API
# -------------------------------------------------------------------
inputs = layers.Input(shape=(32,32,3))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.1)(x)
x = layers.Conv2D(32, (3,3), activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=64,
          validation_data=(x_test, y_test))

# -------------------------------------------------------------------
# 3. Prepare SHAP explainer
# -------------------------------------------------------------------
# Use numpy arrays (not tensors) to avoid KerasTensor warning
background = np.array(x_train[:100])
test_samples = np.array(x_test[:10])

# SHAP now prefers Explainer + GradientExplainer interface
explainer = shap.GradientExplainer(model, background)

# Compute SHAP values
shap_values = explainer.shap_values(test_samples)

# -------------------------------------------------------------------
# 4. Safe visualization
# -------------------------------------------------------------------
# Normalize shap_values to [0,1] for imshow
normed_shap = []
for s in shap_values:
    s_min, s_max = s.min(), s.max()
    normed = (s - s_min) / (s_max - s_min + 1e-8)
    normed_shap.append(normed)

# Visualize
shap.image_plot(normed_shap, test_samples)
plt.show()
