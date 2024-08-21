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
from data_loading import load_data