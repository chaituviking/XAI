import torch
import torch.nn as nn
import torch.nn.functional as F
from train_test import print_train
from weights import ImportanceDependentPerturbationBlock

file_paths = '/home/vardan/xai/datasets/FreqShapeUD/split={}.pt'

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 0 or 1 the input using a threshold of 0.5
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BinaryLayer(nn.Module):
    def forward(self, input):
        return STE.apply(input)

#  usage
input_dim = 50  # Input dimension, should match our data
hidden_dim = 50  # Hidden dimension
activation_fn = torch.tanh  #ReLu,sigmoid,...

model = ImportanceDependentPerturbationBlock(input_dim, hidden_dim, activation_fn)

x, y = print_train(file_paths, file_index=1, batch_number=1)
if x is not None and y is not None:
    print("Returned values:")
    print(f"x: {x}")
    print(f"y: {y}")
else:
    print("Failed to load the particular batch.")

x = x.view(1, -1).float()  # Reshape to match the expected input shape

weights = model(x)
print(weights)

binary_layer = BinaryLayer()

# Apply the STE to the importance weights
binary_weights = binary_layer(weights)

print(binary_weights)
print(binary_weights.shape)
