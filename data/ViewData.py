import numpy as np
import pandas as pd
import torch

bp_model_packet = np.dtype([
        ("ip", 'i8'),
        ("branch_type", 'u1'),
        ("branch_addr", 'i8'),
        ("branch_prediciton", 'u1')
    ])

def read_data(filename):

    with open(filename, "rb") as file:
        b = file.read()

    np_data = np.frombuffer(b, bp_model_packet)
    return pd.DataFrame(np_data)
    

class Dataset(torch.utils.data.Dataset):
    def __init__(self,file_name):
        d = read_data(file_name)
 
        x = d.iloc[:,0:3].values
        y = d.iloc[:,3].values
 
        self.x_train = torch.from_numpy(x)#, dtype=torch.float64)
        self.y_train = torch.from_numpy(y)#, dtype=torch.float64)
 
    def __len__(self):
        return len(self.y_train)
   
    def __getitem__(self, idx):
        return self.x_train[idx],self.y_train[idx]



ds = Dataset("../models/2_bit_sample.bin")
train_loader = torch.utils.data.DataLoader(ds)

for batch, label in train_loader:
    print("batch = {0} and label = {1}".format(batch, label))
