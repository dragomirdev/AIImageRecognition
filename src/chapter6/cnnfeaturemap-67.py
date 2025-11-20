import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

# Reshape: add channel dimension (grayscale = 1)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Normalize (0–255 → 0–1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test))

# Create a model to output intermediate layers
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

# Pass one image
img = x_test[0].reshape(1, 28, 28, 1)
activations = activation_model.predict(img)

# Visualize first layer’s activations
first_layer_activation = activations[0]

plt.figure(figsize=(10, 8))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(first_layer_activation[0, :, :, i], cmap='gray')
    plt.axis('off')
plt.suptitle("First Convolutional Layer Feature Maps")
plt.show()
