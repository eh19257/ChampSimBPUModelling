#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
from datetime import datetime

import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True

DEVICE = None

def parse_args():
    return None


def read_data(filepath):
    bp_model_packet = np.dtype([
        ("ip", np.float64),
        ("branch_type", '<u1'),
        ("branch_addr", np.float64),
        ("branch_prediciton", '<u1')
    ])

    with open(filepath, "rb") as file:
        b = file.read(18)

    np_data = np.frombuffer(b, bp_model_packet)
    return pd.DataFrame(np_data)


class Model(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, num_layers):
        super().__init__()  

        self.num_layers = num_layers            # Number of layers in the RNN
        self.hidden_size = hidden_size          # number of features in the hidden state h

        # RNN componenet
        self.RNN = nn.RNN(in_size, hidden_size, num_layers)

        # Fully connected layer component
        self.fully_connected = nn.Linear(hidden_size, out_size)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        batch_size = x.size(0)

        #Initializing hidden state for first input using method defined below
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)

        # input data & hidden state is given to the RNN
        out, hidden = self.RNN(x, hidden)

        print(out.shape)
        print(out)

        # We now need to crush the output of the RNN so that it can fit into the input of our fully connected layer
        #out = out.contigious().view(-1, self.hidden_size)
        out = out.reshape((-1, self.hidden_size))
        out = self.fully_connected(out)
        #print(out)
        
        return out, hidden


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        d = read_data(file_name)
 
        x = d.iloc[:,0:3].values
        y = d.iloc[:,3].values
 
        self.x_train = torch.tensor(x,dtype=torch.float32)
        self.y_train = torch.tensor(y,dtype=torch.float32)
 
    def __len__(self):
        return len(self.y_train)
   
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer,         #: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                batch = torch.unsqueeze(batch, dim=0).to(self.device)
                labels = torch.unsqueeze(labels, dim=0).to(self.device)
                data_load_end_time = time.time()


                ## TASK 1: Compute the forward pass of the model, print the output shape
                ##         and quit the program
                
                #m = nn.Conv2d(3, 32, (5, 5), padding = 4)
                #output = m(batch)
                #print("batch:",batch)
                #print("label:", labels)
                #print(batch.shape)
                #foo = torch.unsqueeze(batch, dim=0)
                logits = self.model.forward(batch)
                print("logits:", logits)
                print("lablel:", labels)
                #print(output.shape)
                #import sys; sys.exit(1)


                ## TASK 7: Rename `output` to `logits`, remove the output shape printing
                ##         and get rid of the `import sys; sys.exit(1)`

                ## TASK 9: Compute the loss using self.criterion and
                ##         store it in a variable called `loss`
                #loss = torch.tensor(0)

                loss = self.criterion(logits[0], labels)

                ## TASK 10: Compute the backward pass

                loss.backward()

                ## TASK 12: Step the optimizer and then zero out the gradient buffers.

                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits[0].argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            #self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
    

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def main():

    FILENAME = "2_bit_ls_float.bin"
    # Use GPU if their avaliable
    #if torch.cuda.is_available():
    #    DEVICE = torch.device("cuda")
    #else:
    
    DEVICE = torch.device("cpu")


    # Hyper parameters
    LR = 0.01
    MOMENTUM = 0.9
    
    ##############################
    # /     Data loading
    #/

    BATCHSIZE = 5
    WORKER_COUNT = 4

    train_dataset = Dataset(FILENAME)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size = BATCHSIZE,#args.batch_size,
        pin_memory=True,
        num_workers = WORKER_COUNT,#args.worker_count,
    )

    test = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size = BATCHSIZE,#args.batch_size,
        pin_memory=True,
        num_workers = WORKER_COUNT,#args.worker_count,
    )
    in_size = 3#len(train_dataset.x_train)
    out_size = 1
    hidden_size = 1
    num_layers = 1
    
    # Initialise model
    model = Model(in_size, out_size, hidden_size, num_layers)

    # Is this loss? function
    criterion = nn.CrossEntropyLoss()

    # Optimiser      .parameters() returns an iterator of the models parameters - "This is typically passed to an optimizer."
    optimiser = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)


    #############################################################################
    # /     Setup the class "Trainer" - this is used train the actual model     
    #/ 

    # Used for writing logs out
    #summary_writer = SummaryWriter(
    #        str(FILENAME + datetime.today().strftime("%d-%m-%Y") + "--" + datetime.now().strftime("%H-%M-%S")),
    #        flush_secs=5
    #)


    # trainer class - take from COMSM0045: Applied Deep Learning
    trainer = Trainer(model, train_loader, train_loader, criterion, optimiser, None, DEVICE)
    trainer.train(20,
                  2,
                  print_frequency = 10,
                  log_frequency = 10 )

main()