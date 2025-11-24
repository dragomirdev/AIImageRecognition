import tensorflow as tf

# Example: preprocess CIFAR-10 images
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Resize and normalize
x_train = tf.image.resize(x_train, (224, 224)) / 255.0
x_test = tf.image.resize(x_test, (224, 224)) / 255.0