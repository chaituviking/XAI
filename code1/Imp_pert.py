import pandas as pd
import numpy as np
from data_loading import load_data
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearTransform(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()

    def forward(self, x):
        return torch.tanh(x) #ReLu,sigmoid,..

class Softmax(nn.Module):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)
    

class PerturbationModel(nn.Module):
    def __init__(self, t, d):
        super(PerturbationModel, self).__init__()
        self.t = t
        self.d = d
        
        # First path: along the time axis
        self.linear1_1 = LinearTransform(t, d)
        self.linear2_1 = LinearTransform(d, t)
        self.activation1 = Activation()
        self.linear3_1 = LinearTransform(t, t)
        self.softmax1 = Softmax(dim=-1)
        
        # Second path: along the feature axis
        # self.linear1_2 = LinearTransform(t, d)
        # self.linear2_2 = LinearTransform(d, t)
        # self.activation2 = Activation()
        # self.linear3_2 = LinearTransform(t, 1)
        # self.softmax2 = Softmax(dim=1)
        
    def forward(self, x):
        x = x.float()

        x = x.view(x.size(0), -1) # To ensure that the input shape is [batch_size, t]
        
        # First path: along the time axis
        H_txd = self.linear1_1(x)
        Z_txd = self.linear2_1(H_txd)
        Z_prime_txd = self.activation1(Z_txd)
        e_1xt = self.linear3_1(Z_prime_txd)
        a1_1xt = self.softmax1(e_1xt)
        
        # Second path: along the feature axis
        # H_fxd = self.linear1_2(x)
        # Z_fxd = self.linear2_2(H_fxd)
        # Z_prime_fxd = self.activation2(Z_fxd)
        # e_1xf = self.linear3_2(Z_prime_fxd)
        # a2_1xf = self.softmax2(e_1xf)
        
        return a1_1xt


t = 50  # No.of time steps
d = 100  # Intermediate dim

model = PerturbationModel(t, d)
file_paths = '/home/vardan/xai/datasets/FreqShapeUD/split={}.pt'

X_test, Y_test = load_data(file_paths, data_key='test')

X_test = X_test.permute(1, 0, 2)  # Adjust shape to [5000, 50, 1]


a1_1xt = model(X_test) # Get the importance weights

a1_1xt = a1_1xt.unsqueeze(-1)  # Shape: [5000, 50, 1]

print("Importance weights along time axis (a1_1xt):", a1_1xt.shape)
print(a1_1xt[0])
# print("Importance weights along feature axis (a2_1xf):", a2_1xf.shape)
# print(a2_1xf[0])
p=a1_1xt*X_test
print(p[0])
print(p.shape)
print(X_test.shape)