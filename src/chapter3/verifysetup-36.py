
import tensorflow as tf
import torch


print("TensorFlow version:", tf.__version__)
print("PyTorch version:", torch.__version__)

print("Num GPUs Available (TensorFlow):", len(tf.config.list_physical_devices('GPU')))
print("PyTorch CUDA Available:", torch.cuda.is_available())

