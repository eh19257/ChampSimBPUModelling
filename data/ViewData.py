import numpy as np
import pandas as pd
import torch

from torch import nn

import numpy as np

bp_model_packet = np.dtype([
        ("ip", 'i8'),
        ("branch_type", 'u1'),
        ("branch_addr", 'i8'),
        ("branch_prediciton", 'u1')
    ])


DEVICE = torch.device("cpu")

class Model(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, num_layers):
        super().__init__()  

        self.num_layers = num_layers            # Number of layers in the RNN
        self.hidden_size = hidden_size          # number of features in the hidden state h

        # RNN componenet
        self.RNN = nn.RNN(in_size, hidden_size, num_layers)

        # Fully connected layer component
        self.full_connected = nn.Linear(hidden_size, out_size)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)

        # input data & hidden state is given to the RNN
        out, hidden = self.RNN(x, hidden)

        # We now need to crush the output of the RNN so that it can fit into the input of our fully connected layer
        out = out.contigious().view(-1, self.hidden_size)
        out = self.fully_connected(out)
        
        return out, hidden


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
train_loader = torch.utils.data.DataLoader(ds, batch_size=1, )

for batch, label in train_loader:
    #print(batch.shape)#"batch = {0} and label = {1}".format(batch, label)
    print("batch:", batch)
    foo = torch.unsqueeze(batch, dim=0)
    print(foo)
    print("\n\n")

#print(train_loader.shape)

'''

in_size = 3#len(train_dataset.x_train)
out_size = 1
hidden_size = 1
num_layers = 1

# Initialise model
model = Model(in_size, out_size, hidden_size, num_layers)



# Define hyperparameters
n_epochs = 100
lr=0.01

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

input_seq = ds.x_train.to(DEVICE)#.input_seq.to(device)
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    #input_seq = input_seq.to(device)
    output, hidden = model(input_seq)
    output = output.to(device)
    target_seq = target_seq.to(device)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))


'''