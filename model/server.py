from model import MODEL

latent_dims = 5
import numpy as np

def call():
    v = MODEL(latent_dims)
    vec = np.random.rand(latent_dims)
    v.reconstruct(vec)
