import torch

from src.deep_learning.dataset import ExoVisionDataset
from src.deep_learning.cnn_model import ExoVisionCNN


# Load dataset
dataset = ExoVisionDataset()

# Get one sample
x, y = dataset[0]

# Add batch dimension
x = x.unsqueeze(0)

print("Input shape:")
print(x.shape)


# Create model
model = ExoVisionCNN()

print("\nModel:")
print(model)


# Forward pass
output = model(x)

print("\nOutput shape:")
print(output.shape)

print("\nRaw output:")
print(output)