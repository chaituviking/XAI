import sys
sys.path.append('/home/vardan/xai/')
import torch

file_paths = '/home/vardan/xai/datasets/FreqShapeUD/split={}.pt'

def print_train_loader(file_paths, start_batch=None, end_batch=None):
    for i in range(1, 6):
        file_path = file_paths.format(i)
        loaded_dict = torch.load(file_path)
        print(f"Contents of 'train_loader' in file {i}:")

        train_loader = loaded_dict.get('train_loader')
        if train_loader:
            for j, item in enumerate(train_loader):
                # Print batches only within the specified range
                if start_batch is not None and end_batch is not None:
                    if j < start_batch or j > end_batch:
                        continue

                print(f"  Batch {j}: {item}")
                if isinstance(item, (tuple, list)):
                    for k, el in enumerate(item):
                        if isinstance(el, torch.Tensor):
                            print(f"    Element {k}: Tensor with shape {el.shape}")
                        else:
                            print(f"    Element {k}: {el}")
        else:
            print("  'train_loader' not found in the file.")
        print(item[0].shape)
        print(item[1].shape)
        print(item[2].shape)


def load_and_print_test_data(file_paths, key='test'):
    for i in range(1, 6):
        file_path = file_paths.format(i)
        loaded_dict = torch.load(file_path)
        
        if key in loaded_dict:
            print(f"Contents of '{key}' in file {i}:")
            value = loaded_dict[key]
            
            if isinstance(value, torch.utils.data.DataLoader):
                for j, batch in enumerate(value):
                    print(f"  Batch {j}:")
                    if isinstance(batch, (tuple, list)) and all(isinstance(el, torch.Tensor) for el in batch):
                        print(f"    Shapes: {[el.shape for el in batch]}")
                        print(f"    Data: {batch}")
                    else:
                        print(f"    {batch}")
            else:
                print(f"  {value}")
        else:
            print(f"Key '{key}' not found in file {i}")
        print(value[0].shape)
        print(value[1].shape)
        print(value[2].shape)

# Usage
# load_and_print_test_data(file_paths)

def load_and_print_val_data(file_paths, key='val'):
    for i in range(1, 6):
        file_path = file_paths.format(i)
        loaded_dict = torch.load(file_path)
        
        if key in loaded_dict:
            print(f"Contents of '{key}' in file {i}:")
            value = loaded_dict[key]
            
            if isinstance(value, torch.utils.data.DataLoader):
                for j, batch in enumerate(value):
                    print(f"  Batch {j}:")
                    if isinstance(batch, (tuple, list)) and all(isinstance(el, torch.Tensor) for el in batch):
                        print(f"    Shapes: {[el.shape for el in batch]}")
                        print(f"    Data: {batch}")
                    else:
                        print(f"    {batch}")
            else:
                print(f"  {value}")
        else:
            print(f"Key '{key}' not found in file {i}")
        print(value[0].shape)
        print(value[1].shape)
        print(value[2].shape)

load_and_print_test_data(file_paths)
# print_train_loader(file_paths, start_batch=1,end_batch=1)
