import torch; torch.manual_seed(0)
import torch.nn as nn
import numpy as np

latent_dims = 5

device= 'cpu'

class VariationalEncoder (nn.Module):
    def __init__(self, latent_dims, device):
        super(VariationalEncoder, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(800, 512)
        self.fc2 = nn.Linear(512, latent_dims)
        self.fc3 = nn.Linear(512, latent_dims)
        
        self.relu = nn.ReLU()
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.device)
        self.N.scale = self.N.scale.to(self.device)
        self.kl = 0
        
        
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        mu =  self.fc2(x)
        sigma = torch.exp(self.fc3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims, device):
        super(Decoder, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(latent_dims, 512)
        self.fc2 = nn.Linear(512, 800)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.sigmoid(self.fc2(z))
        z = torch.reshape(z, (-1, 400, 2))
        return z

class VAE(nn.Module):
    def __init__(self, latent_dims, device):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, device)
        self.decoder = Decoder(latent_dims, device)
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class MODEL:
    def __init__(self, latent_dims):
        super(MODEL, self).__init__()
        self.latent_dims = latent_dims
        self.model = VAE(latent_dims, device)
        model_path = './model.pt'
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
# import sys; sys.path.append("/home/archo/vae_dr/dr-vae-backend/model"); from model import VAE_MODEL; v = VAE_MODEL(5)
    
    def reconstruct(self, latent_vec):
        z = torch.Tensor([latent_vec]).to(device)
        x_hat = self.model.decoder(z)
        return x_hat