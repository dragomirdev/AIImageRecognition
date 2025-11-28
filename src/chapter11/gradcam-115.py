from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# 1. Load CIFAR-10
# -------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype(np.float32)/255.0, x_test.astype(np.float32)/255.0
y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# -------------------------------------------------------------------
# 2. Build model using Functional API
# -------------------------------------------------------------------
inputs = layers.Input(shape=(32,32,3))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.1)(x)

x = layers.Conv2D(32, (3,3), activation='relu', name='conv2d_0')(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(64, (3,3), activation='relu', name='conv2d_1')(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.summary()

# -------------------------------------------------------------------
# 3. Compile & train (short run)
# -------------------------------------------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=64,
          validation_data=(x_test, y_test))

# -------------------------------------------------------------------
# 4. Grad-CAM function
# -------------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# -------------------------------------------------------------------
# 5. Grad-CAM example
# -------------------------------------------------------------------
# Load a test image
img = tf.keras.utils.load_img('dog.png', target_size=(32, 32))
img_array = np.expand_dims(tf.keras.utils.img_to_array(img)/255.0, axis=0)

# Find last Conv2D layer automatically
last_conv_layer_name = [l.name for l in model.layers if isinstance(l, layers.Conv2D)][-1]
print("🎯 Using layer:", last_conv_layer_name)

# Generate Grad-CAM heatmap
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# Resize + overlay using PIL + matplotlib (no cv2)
heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((224, 224))
heatmap_color = plt.cm.jet(np.array(heatmap_img) / 255.0)[:, :, :3]

original_img = np.array(img.resize((224, 224))) / 255.0
superimposed = np.clip(0.6 * original_img + 0.4 * heatmap_color, 0, 1)

plt.figure(figsize=(6, 6))
plt.imshow(superimposed)
plt.axis('off')
plt.title("Grad-CAM Visualization (No cv2)")
plt.show()
