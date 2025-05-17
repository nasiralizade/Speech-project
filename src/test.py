import torch
import torch.nn as nn

cnn = nn.Sequential(
    nn.Conv2d(2, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten()
)

# Example input tensor: (batch_size=1, channels=2, height=100, width=40)
example_input = torch.randn(1, 2, 100, 40)
output = cnn(example_input)

print("CNN output shape:", output.shape)