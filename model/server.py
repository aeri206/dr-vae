from model import MODEL

latent_dims = 5
import numpy as np
import time

def call():
    v = MODEL(latent_dims)
    vec = np.random.rand(latent_dims)
    start_time = time.time()
    v.reconstruct(vec)
    print("--- %s seconds ---" % (time.time() - start_time))

