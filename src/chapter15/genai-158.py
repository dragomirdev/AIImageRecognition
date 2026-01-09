# ------------------------------------------------------------
# Deep Convolutional GAN (DCGAN) – Generate Faces (Local CelebA)
# ------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# 1. Parameters
# ------------------------------------------------------------
IMG_SIZE = 64
BATCH_SIZE = 32
LATENT_DIM = 100
EPOCHS = 50

# ------------------------------------------------------------
# 2. Data Loading and Preprocessing
# ------------------------------------------------------------
def preprocess(example):
    """Handles dicts, tuples, and plain image tensors safely."""
    if isinstance(example, dict):
        image = example.get("image", None)
        if image is None:
            raise ValueError("Dictionary example missing 'image' key.")
    elif isinstance(example, tuple):
        image = example[0]  # ignore labels
    else:
        image = example

    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5  # normalize to [-1, 1]
    return image


def load_local_celeba():
    """Loads CelebA-like images from local directory."""
    data_dir = os.path.expanduser("img_align_celeba")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Directory not found: {data_dir}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode=None  # no labels for GAN
    )

    dataset = dataset.map(preprocess).shuffle(100).repeat().prefetch(tf.data.AUTOTUNE)
    print("✅ Loaded local CelebA images successfully.")
    return dataset


dataset = load_local_celeba()

# Steps per epoch for a small local dataset
num_images = len(os.listdir("img_align_celeba"))
steps_per_epoch = max(1, num_images // BATCH_SIZE)

for batch in dataset.take(1):
    print("🎨 Batch shape:", batch.shape)

# ------------------------------------------------------------
# 3. Generator Model
# ------------------------------------------------------------
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8*8*512, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((8, 8, 512)),
        layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, (5,5), strides=(1,1), padding='same', use_bias=False, activation='tanh')
    ])
    return model

generator = build_generator()
generator.summary()

# ------------------------------------------------------------
# 4. Discriminator Model
# ------------------------------------------------------------
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[IMG_SIZE, IMG_SIZE, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(256, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

discriminator = build_discriminator()
discriminator.summary()

# ------------------------------------------------------------
# 5. Losses and Optimizers
# ------------------------------------------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
disc_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

# ------------------------------------------------------------
# 6. Training Step
# ------------------------------------------------------------
@tf.function
def train_step(images):
    noise = tf.random.normal([tf.shape(images)[0], LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = (cross_entropy(tf.ones_like(real_output), real_output) +
                     cross_entropy(tf.zeros_like(fake_output), fake_output)) / 2

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

    return gen_loss, disc_loss

# ------------------------------------------------------------
# 7. Visualization Helper
# ------------------------------------------------------------
os.makedirs("generated_faces", exist_ok=True)
seed = tf.random.normal([16, LATENT_DIM])

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i] * 0.5 + 0.5).numpy())
        plt.axis('off')
    plt.suptitle(f"Epoch {epoch}")
    fig.savefig(f"generated_faces/epoch_{epoch:03d}.png")
    plt.close(fig)

# ------------------------------------------------------------
# 8. Training Loop
# ------------------------------------------------------------
for epoch in range(EPOCHS):
    for step, image_batch in enumerate(dataset.take(steps_per_epoch)):
        g_loss, d_loss = train_step(image_batch)

    print(f"Epoch {epoch+1}/{EPOCHS} | Generator Loss: {g_loss:.4f} | Discriminator Loss: {d_loss:.4f}")
    generate_and_save_images(generator, epoch+1, seed)

print("✅ Training complete! Generated images saved in /generated_faces")
