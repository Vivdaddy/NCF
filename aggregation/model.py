
# In[1]
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import os

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers=1, batch_size = 1024, device="cpu"):
        super(LSTM, self).__init__()

        self.device = device
        self.batch_size = batch_size
        
        # Defining some parameters (hidden_dem, n_layers) = (hidden dimension of LSTM, num of layers in LSTM)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # LSTM Layer: (output_size, input_size) = (num of items, item embedding length)
        self.LSTM = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.LSTM(x, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        inp = out[:, -1, :].contiguous().view(-1, self.hidden_dim)
        out = self.fc(inp)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device).detach(), torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device).detach())
        return hidden


# MLP: adjust the number of layers based on input and output size
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        hidden_size = np.power(2, int(np.log2(input_size)))
        modules = []
        if hidden_size < output_size:
            modules.append(nn.Linear(input_size, hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_size, output_size))

        else:
            hidden_layer = [input_size, hidden_size]
            while hidden_size > output_size*2:
                hidden_size = hidden_size//2
                hidden_layer.append(hidden_size)
            hidden_layer.append(output_size)

            for i in range(len(hidden_layer)-1):
                modules.append(nn.Linear(hidden_layer[i], hidden_layer[i+1]))
                modules.append(nn.ReLU())
            modules = modules[:-1]

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


# MLP_item: adjust the number of layers based on input. shrink to scalar node (1-by-1)
class MLP_ITEM(nn.Module):
    def __init__(self, input_size, output_size = 1, initial = None, bias = True):
        super(MLP_ITEM, self).__init__()
        self.bias = bias

        # (input_size, output_size) = (num of ensembles, 1)
        hidden_size = np.power(2, int(np.log2(input_size)))//2 # 1 > 2
        modules = []
        modules.append(nn.Linear(input_size, 4, bias = bias))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(4, 1, bias = bias))


        # hidden_layer = [input_size, hidden_size]

        # while hidden_size > 1:
        #     hidden_size = hidden_size//4 # 2 > 4
        #     if hidden_size == 0:
        #         hidden_layer.append(1)
        #         continue
        #     hidden_layer.append(hidden_size)
        
        # for i in range(len(hidden_layer)-1):
        #     modules.append(nn.Linear(hidden_layer[i], hidden_layer[i+1], bias = bias))
        #     modules.append(nn.ReLU())
        # modules = modules[:-1]

        self.layers = nn.Sequential(*modules)
        
        # initialization: default == None
        if initial == initial:
            self.layers.apply(self.init_weights)

    def init_weights(self, hidden_layer):
        # initialize weight = 1, bias = 0 to approx unweighted average
        if isinstance(hidden_layer, nn.Linear):
            hidden_layer.weight.data.fill_(1)
            if self.bias == True:
                hidden_layer.bias.data.fill_(0)

    def forward(self, x):
        return self.layers(x)


