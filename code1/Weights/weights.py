# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# df= pd.read_csv("/home/vardan/xai/datasets/Boiler/full.csv")
# scaler = StandardScaler()
# data_normalized = scaler.fit_transform(df)
# print(data_normalized)



import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_test import print_train

file_paths = '/home/vardan/xai/datasets/FreqShapeUD/split={}.pt'


class ImportanceDependentPerturbationBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation_fn=torch.tanh):
        super(ImportanceDependentPerturbationBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        
        # Define layers for the upper path
        self.linear_transform1_upper = nn.Linear(input_dim, hidden_dim)
        self.linear_transform2_upper = nn.Linear(hidden_dim, hidden_dim)
        self.linear_transform3_upper = nn.Linear(hidden_dim, hidden_dim)
        self.linear_transform4_upper = nn.Linear(hidden_dim, input_dim)
        
        # Define layers for the lower path
        self.linear_transform1_lower = nn.Linear(input_dim, hidden_dim)
        self.linear_transform2_lower = nn.Linear(hidden_dim, hidden_dim)
        self.linear_transform3_lower = nn.Linear(hidden_dim, hidden_dim)
        self.linear_transform4_lower = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Upper path(time steps)
        H_t_d_upper = self.linear_transform1_upper(x)
        Z_t_d_upper = self.linear_transform2_upper(H_t_d_upper)
        Z_prime_t_d_upper = self.activation_fn(Z_t_d_upper)
        e_1_t_upper = self.linear_transform3_upper(Z_prime_t_d_upper)
        a1_1_t_upper = F.softmax(e_1_t_upper, dim=-1)
        
        # Lower path(features) only use for multivariant
        # H_f_x_t_lower = self.linear_transform1_lower(x)
        # Z_f_x_d_lower = self.linear_transform2_lower(H_f_x_t_lower)
        # Z_prime_f_x_d_lower = self.activation_fn(Z_f_x_d_lower)
        # e_1_f_lower = self.linear_transform3_lower(Z_prime_f_x_d_lower)
        # a2_1_f_lower = F.softmax(e_1_f_lower, dim=-1)
        
        #weights cal
        a_prime_f_x_t = x + a1_1_t_upper
        weights= a_prime_f_x_t * x
        
        return weights

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

output = model(x)
print("Imp weights:")
print(output)
print(output.shape)
print(x.shape)
print(x)
