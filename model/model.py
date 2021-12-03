import torch; torch.manual_seed(0)
import torch.nn as nn
import numpy as np
import math
from collections import Counter
import json

device= 'cpu'
    
class VAE(nn.Module):
    def __init__(self, latent_dims, point_count, hidden_dims: list = None):
        super(VAE, self).__init__()
        self.latent_dims = latent_dims
        self.point_count = point_count
        self.hidden_dims = hidden_dims
        en_layers = []
        in_dim = self.point_count * 2
        for h_dim in hidden_dims:
            en_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU()
                )
            )
            in_dim = h_dim
        self.encoder = nn.Sequential(*en_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dims)
        self.fc_sigma = nn.Linear(hidden_dims[-1], latent_dims)
        
        de_layers = []
        hidden_dims = hidden_dims[::-1]
        in_dim = self.latent_dims
        for h_dim in hidden_dims:
            de_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU()
                )
            )
            in_dim = h_dim
        de_layers.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1], self.point_count * 2),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*de_layers)
        
    def encode(self, x): 
        x = self.encoder(torch.flatten(x, start_dim = 1))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma
    
    def reparameterize(self, mu, sigma):
        std = torch.exp(sigma * 0.5)
        eps = torch.randn_like(std)
        res = eps * std + mu
        return res
    
    def decode(self, z):
        z = self.decoder(z)
        return torch.reshape(z, (-1, self.point_count, 2))
    
    def forward(self, x):
        # x.shape : torch.Size([batch_size, 2, point_count])
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        return self.decode(z), mu, sigma
        
    def get_z(self, x): #x.shape : (1, 2 * point_count)
        return self.reparameterize(*(self.encode(x)))
    
    def get_recon_x(self, z):
        return self.decode(z)
        
    


class MODEL:
    def __init__(self, model_path, point_num):
        super(MODEL, self).__init__()
        
## 여기부터
        all_dims = []
        h_dims = []
        with open(model_path + 'training_info.json') as f:
            info = (json.load(f))
            h_dims = info["hidden_dims"]
            all_dims = h_dims[:]
            self.latent_dims = info["latent_dims"]
            all_dims.append(self.latent_dims)
            all_dims.append(2 * point_num)
            all_dims.sort()
        
        h_model_path = model_path + 'model.pt'
        h_model_state = torch.load(h_model_path, map_location=device)

        self.v = VAE(latent_dims = self.latent_dims,
                    point_count = point_num,
                    hidden_dims = h_dims).to(device)
        self.v.load_state_dict(h_model_state, strict=False)
        
        
    
    
    def reconstruct(self, latent_vec):
        z = torch.Tensor(latent_vec).view(-1, len(latent_vec)).to(device)
        x_hat = self.v.get_recon_x(z)
        return x_hat

        