
# Project Title

Explainability for anamoly detection in time series data


## Datasets
We using the same Datasets used by [TimeX](https://github.com/mims-harvard/TimeX) and [TimeX++](https://github.com/zichuan-liu/TimeXplusplus)
## File Organization

train_test.py: Contains functions for loading and printing data batches from the saved datasets.

Weights.py: Defines the neural network model for importance-dependent perturbation block.

STE.py: The main file that integrates the model with data and applies binary thresholding(either 0 or 1) to the computed importance weights.
## File Descriptions and Functionality

1. **train_test.py**
Purpose: This file is responsible for loading and displaying specific data batches from pre-processed datasets.

Functions:

 **print_train**(file_paths, file_index, batch_number):
- Input:
    - file_paths (str): The paths of dataset files.
    - file_index (int): The specific index of the dataset file to be loaded.
    - batch_number (int): The specific batch number to be recover from the dataset.

Output:
- x: The feature tensor for the batch.
- y: The target tensor for the batch.
Task: Loads a specific batch from the dataset files and returns the feature and target tensors. If the batch or file is not found, it returns None.

2. **weights.py**
Purpose: This file defines the ImportanceDependentPerturbationBlock class, which implements a flow of linear neural network layers with series of activation function and softmax layer for calculating importance weights based on input data.

Classes:
ImportanceDependentPerturbationBlock(nn.Module):

Attributes:
- input_dim (int): The dimension of the input data.
- hidden_dim (int): The dimension of the hidden layers.
- activation_fn (torch function): The activation function to be used in the layers.(Ex:tanh,ReLu,Sigmoid,...)
Methods:
__init__(self, input_dim, hidden_dim, activation_fn=torch.tanh): Initializes the neural network with specified input and hidden dimensions and an activation function.
- forward(self, x):
Input:
- x: The input tensor.
Output:
- weights: The calculated importance weights based on the input.
Task: Passes the input through a series of linear transformations and activation functions to compute importance weights for the input features.

3. **STE.py**
Purpose: This file combines the functionality of the previous two files to load data, compute importance weights, and apply a binary threshold (either 0 or 1) to the weights using a custom binary layer.

Classes:

STE(torch.autograd.Function):
Methods:
- forward(ctx, input): Binarizes the input based on a threshold of 0.5(you can change this Ex: 0.3,0.4,...).
- backward(ctx, grad_output): Passes through the gradient for backpropagation.
BinaryLayer(nn.Module):
Methods:
- forward(self, input): Applies the STE function to the input to generate binary weights.

## Workflow
- Load a specific batch of data using print_train.
- Initialize an instance of the ImportanceDependentPerturbationBlock model.
- Compute the importance weights using the model.
- Apply the binary threshold to the importance weights using BinaryLayer.
## Usage/Examples
The main.py file demonstrates how to use the components of this project:

- Load data from the dataset using print_train.
- Initialize the model with the required input and hidden dimensions.
- Compute importance weights from the input data.
- Apply a binary threshold to the computed weights (0 or 1).

```javascript
# Example usage in STE.py
x, y = print_train(file_paths, file_index=1, batch_number=1)
if x is not None and y is not None:
    x = x.view(1, -1).float()  # Reshape to match the expected input shape

    model = ImportanceDependentPerturbationBlock(input_dim=50, hidden_dim=50, activation_fn=torch.tanh)
    weights = model(x)
    
    binary_layer = BinaryLayer()
    binary_weights = binary_layer(weights)

    print(binary_weights)

```


## Authors

- [@chaituviking](/https://github.com/chaituviking)


## Installation

Install all packages

```bash
  pip install torch numpy pandas
```
    
## How to reproduce the original results of TimeX++
You can find Reproducibility [here](https://github.com/zichuan-liu/TimeXplusplus#how-to-run)
##Screenshot of important weights
![photo_2024-08-11_16-44-54](https://github.com/user-attachments/assets/c41037de-c1ee-4555-bc8b-8d7dd5fed3bd)
