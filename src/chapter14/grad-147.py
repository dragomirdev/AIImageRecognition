import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# -------------------------------------------------------------
# 1. Load datasets
# -------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    "chest_xray/train", image_size=(224, 224), batch_size=32
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "chest_xray/val", image_size=(224, 224), batch_size=32
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "chest_xray/test", image_size=(224, 224), batch_size=32
)

# Normalize + augment
normalization = layers.Rescaling(1./255)
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

train_ds = train_ds.map(lambda x, y: (augmentation(normalization(x)), y))
val_ds   = val_ds.map(lambda x, y: (normalization(x), y))
test_ds  = test_ds.map(lambda x, y: (normalization(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# -------------------------------------------------------------
# 2. Functional CNN model
# -------------------------------------------------------------
inputs = layers.Input(shape=(224, 224, 3))

x = layers.Conv2D(32, (3,3), activation='relu', name="conv2d_1")(inputs)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(64, (3,3), activation='relu', name="conv2d_2")(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(128, (3,3), activation='relu', name="conv2d_3")(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid', name="output")(x)

model = models.Model(inputs, outputs, name="ChestXrayCNN")

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------------------------------------
# 3. Train
# -------------------------------------------------------------
history = model.fit(train_ds, validation_data=val_ds, epochs=3)

# -------------------------------------------------------------
# 4. Confusion matrix
# -------------------------------------------------------------
y_true = np.concatenate([y for _, y in test_ds], axis=0)
y_pred_probs = np.concatenate([model.predict(x) for x, _ in test_ds], axis=0)
y_pred = (y_pred_probs > 0.5).astype(int)

labels = ["NORMAL", "PNEUMONIA"]
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))

# -------------------------------------------------------------
# 5. Grad-CAM (Functional models work natively)
# -------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_out), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap

# -------------------------------------------------------------
# 6. Grad-CAM example (Side-by-side version)
# -------------------------------------------------------------
img_path = "chest_xray/test/NORMAL/IM-0111-0001.jpeg"  # change to any test image
img = tf.keras.utils.load_img(img_path, target_size=(224,224))
img_array = np.expand_dims(tf.keras.utils.img_to_array(img)/255.0, axis=0)

# Predict
pred = model.predict(img_array)[0][0]
pred_label = "PNEUMONIA" if pred > 0.5 else "NORMAL"

# Find last conv layer
last_conv = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
heatmap = make_gradcam_heatmap(img_array, model, last_conv)

# ✅ Resize heatmap to image size
heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (224, 224)).numpy()
heatmap_resized = np.squeeze(heatmap_resized)

# ✅ Apply color map
heatmap_color = plt.cm.jet(heatmap_resized)[:, :, :3]

# ✅ Blend with original image
original_img = np.array(img) / 255.0
superimposed = np.clip(0.6 * original_img + 0.4 * heatmap_color, 0, 1)

# ✅ Display side-by-side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title("Original Chest X-Ray")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(superimposed)
plt.title(f"Grad-CAM Heatmap\nPrediction: {pred_label} ({pred:.2f})")
plt.axis('off')

plt.tight_layout()
plt.show()

