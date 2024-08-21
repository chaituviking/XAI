import sys
sys.path.append('/home/vardan/xai/')
import torch

file_paths = '/home/vardan/xai/datasets/FreqShapeUD/split={}.pt'
import sys
sys.path.append('/home/vardan/xai/')
import torch

file_paths = '/home/vardan/xai/datasets/FreqShapeUD/split={}.pt'

# def print_train(file_paths, file_index, batch_number):
#     file_path = file_paths.format(file_index)
#     loaded_dict = torch.load(file_path)
#     print(f"Contents of 'train_loader' in file {file_index}:")

#     train_loader = loaded_dict.get('train_loader')
#     if train_loader:
#         for j, item in enumerate(train_loader):
#             if j == batch_number:
#                 x = item[0]  # Element 1
#                 y = item[2]  # Element 2
#                 print(f"  Batch {j}:")
#                 print(f"    x: Tensor with shape {x.shape}\n{x}")
#                 print(f"    y: Tensor with shape {y.shape}\n{y}")
#                 break
#     else:
#         print("  'train_loader' not found in the file.")

    # print(item[0].shape)
    # print(item[1].shape)
    # print(item[2].shape)


import sys
sys.path.append('/home/vardan/xai/')
import torch

file_paths = '/home/vardan/xai/datasets/FreqShapeUD/split={}.pt'

def print_train(file_paths, file_index, batch_number):
    file_path = file_paths.format(file_index)
    loaded_dict = torch.load(file_path)
    print(f"Contents of 'train_loader' in file {file_index}:")

    train_loader = loaded_dict.get('train_loader')
    if train_loader:
        for j, item in enumerate(train_loader):
            if j == batch_number:
                x = item[0]  # Element 1
                y = item[2]  # Element 2
                print(f"  Batch {j}:")
                # print(f"    x: Tensor with shape {x.shape}\n{x}")
                # print(f"    y: Tensor with shape {y.shape}\n{y}")
                return x, y
    else:
        print("  'train_loader' not found in the file.")
        return None, None

# Example usage
# x, y = print_train(file_paths, file_index=1, batch_number=1)
# if x is not None and y is not None:
#     print("Returned values:")
#     print(f"x: {x}")
#     print(f"y: {y}")
# else:
#     print("Failed to load the specified batch.")



