from pathlib import Path

import umap
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding

import numpy as np
import json

def umap_generate_embeddings(raw_data, path_to_save, iter):
	umap_path = path_to_save + "umap/"
	Path(umap_path).mkdir(parents=True, exist_ok=True)
	for i in range(iter):
		min_dist = np.random.rand()
		n_neighbors = np.random.randint(0, 100)
		emb = umap_embedding(raw_data, min_dist, n_neighbors)
		with open(umap_path + str(i) + ".json", "w") as outfile:
			json.dump({
				"min_dist" : min_dist,
				"n_neighbors" : n_neighbors,
				"emb": emb
			}, outfile)




def umap_embedding(raw_data, min_dist, n_neighbors):
	umap_instance = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
	emb_data = umap_instance.fit_transform(raw_data)
	emb_data = emb_data.tolist()
	return emb_data

def tsne_embedding(raw_data, perplexity):
	emb_data = TSNE(n_components=2, metric="euclidean", perplexity=perplexity, random_state=1).fit_transform(raw_data)
	emb_data = emb_data.tolist()
	return emb_data

def isomap_embedding(raw_data, n_neighbors):
	emb_data = Isomap(n_neighbors=n_neighbors).fit_transform(raw_data)
	emb_data = emb_data.tolist()
	return emb_data


def lle_embedding(raw_data):
	lle = LocallyLinearEmbedding()
	emb_data = lle.fit_transform(raw_data)
	return emb_data.tolist()

def densmap_embedding(raw_data, min_dist, n_neighbors):
	umap_instance = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, densmap=True)
	emb_data = umap_instance.fit_transform(raw_data)
	emb_data = emb_data.tolist()
	return emb_data