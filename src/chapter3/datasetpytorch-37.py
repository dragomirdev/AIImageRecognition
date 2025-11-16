from torchvision import datasets, transforms

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
print("Training samples:", len(train_data))