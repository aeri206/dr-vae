## make 2-d embedding of latent values with dr
import numpy as np
import umap
import json

with open('../latent_vector.json') as f:
		data = json.load(f)
		latent_embedding = umap.UMAP().fit_transform(data)

	
		latent_embedding[:,0] = (latent_embedding[:,0] - np.min(latent_embedding[:,0])) / (np.max(latent_embedding[:,0]) - np.min(latent_embedding[:,0]))
		latent_embedding[:,1] = (latent_embedding[:,1] - np.min(latent_embedding[:,1])) / (np.max(latent_embedding[:,1]) - np.min(latent_embedding[:,1]))
		latent_embedding = (latent_embedding * 1.8) - 0.9
		with open('../latent_emb.json', 'w') as json_file:
 				json.dump(latent_embedding.tolist(), json_file)



