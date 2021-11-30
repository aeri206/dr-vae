import torch; torch.manual_seed(0)
import torch.nn as nn
import numpy as np
import math
from collections import Counter


device= 'cpu'
    

class MODEL:
    def __init__(self, model_path, dataset, point_num):
        super(MODEL, self).__init__()
        model_state = torch.load(model_path, map_location=device)
        
        
        count_dims = Counter(model_state[x].shape[0] for x in model_state.keys() if x.endswith('.bias'))
        dims = list(set(model_state[x].shape[0] for x in model_state.keys() if x.endswith('.bias')))
        
        
        decoder = []
        dims.sort()
        if count_dims[2 * point_num] > 2:
            dims.append(2 * point_num)
        self.latent_dims = dims[0]
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        for i in range(0, len(dims)-1):
            fc = nn.Linear(dims[i], dims[i+1])
            ord = str(i+len(dims))
            fc.weight = nn.Parameter(model_state["fc"+ord+".weight"].to(device))
            fc.bias = nn.Parameter(model_state["fc"+ord+".bias"].to(device))
            decoder.append(fc)
        
        self.depth = len(decoder) - 1
        self.decoder = decoder
        
        
    def reconstruct(self, latent_vec):
        
        z = torch.Tensor(latent_vec).view(-1, len(latent_vec)).to(device)
        for i in range(self.depth):
            layer = self.decoder[i]
            z = self.relu(layer(z))
        
        layer = self.decoder[self.depth]
        z = self.sigmoid(layer(z))
        z = torch.reshape(z, (-1, int(z.shape[1] / 2), 2))
        return z