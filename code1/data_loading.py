import sys
sys.path.append('/home/vardan/xai/')
import matplotlib.pyplot as plt
import torch
import seaborn as sns


file_paths = '/home/vardan/xai/datasets/FreqShapeUD/split={}.pt'

def load_data(file_paths, data_key='data'):
    time_series_list = []
    labels_list = []  

    for i in range(1, 6):
        file_path = file_paths.format(i)
        loaded_dict = torch.load(file_path)
        # print(f"File {i} contents: {loaded_dict.keys()}")
        
        time_series = loaded_dict[data_key]

        if data_key == 'train_loader':
            train_data = loaded_dict[data_key]
            for item in train_data:
                if isinstance(item, (tuple, list)):
                    time_series = item[0]
                    labels = item[2] if len(item) > 2 else None
                else:
                    time_series = item
                    labels = None
                time_series_list.append(time_series)
                if labels is not None:
                    labels_list.append(labels)
        else:
            if isinstance(time_series, tuple):
                time_series, labels = time_series[0], time_series[2] if len(time_series) > 2 else None
            else:
                labels = None
            time_series_list.append(time_series)
            if labels is not None:
                labels_list.append(labels)

    X = torch.cat(time_series_list, dim=1)  
    if labels_list:
        Y = torch.cat(labels_list, dim=0)  
        return X, Y
    return X

# X_test,Y_test = load_data(file_paths, data_key='test')
# print("Concatenated test data Y_test:\n", Y_test[0])
# print("Concatenated test data X_test:\n", X_test[0])
# print(Y_test.shape)
# print(X_test.shape)


# if Y_test is not None:
    # print("Concatenated test labels Y_test:\n", Y_test)
    # print(Y_test.shape)

# def plot_time_series_with_labels(time_series, labels, num_series=5):
#     plt.figure(figsize=(15, num_series * 3))

#     for i in range(num_series):
#         plt.subplot(num_series, 1, i+1)
#         plt.plot(time_series[i], label=f'Time Series {i}')
        
#         # Show the class label for each time series
#         plt.title(f'Time Series {i} - Class {labels[i]}')
#         plt.xlabel('Time Steps')
#         plt.ylabel('Value')
#         plt.legend()
    
#     plt.tight_layout()
#     plt.savefig('ts.png')

# # Convert to NumPy array for easier plotting
# time_series_data = X_test.squeeze().numpy()
# labels_data = Y_test.numpy() if Y_test is not None else None

# # Plot the first 5 time series with their labels
# plot_time_series_with_labels(time_series_data, labels_data, num_series=5)

# def plot_label_distribution(labels):
#     plt.figure(figsize=(10, 6))
#     sns.histplot(labels.flatten(), bins=50, kde=True)
#     plt.title('Label Distribution')
#     plt.xlabel('Label Value')
#     plt.ylabel('Frequency')
#     plt.savefig('lb.png')

# # Assuming labels are provided
# if labels_data is not None:
#     plot_label_distribution(labels_data)

















