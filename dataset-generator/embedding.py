from pathlib import Path

import umap
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding

import numpy as np
import json

'''
UMAP
'''
def umap_generate_embeddings(raw_data, path_to_save, iteration, log_interval):
	umap_path = path_to_save + "umap/"
	Path(umap_path).mkdir(parents=True, exist_ok=True)
	for i in range(1, iteration + 1):
		min_dist = np.random.rand()
		n_neighbors = np.random.randint(3, 101)
		emb = umap_embedding(raw_data, min_dist, n_neighbors)
		with open(umap_path + str(i) + ".json", "w") as outfile:
			json.dump({
				"min_dist" : min_dist,
				"n_neighbors" : n_neighbors,
				"emb": emb
			}, outfile)
		if i % log_interval == 0:
			print("UMAP embedding generation: " + str(i) + "/" + str(iteration) + " finished")
	print("*** UMAP embedding generation finished. ***")


def umap_embedding(raw_data, min_dist, n_neighbors):
	umap_instance = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
	emb_data = umap_instance.fit_transform(raw_data)
	emb_data = emb_data.tolist()
	return emb_data

'''
TSNE
'''
def tsne_generate_embeddings(raw_data, path_to_save, iteration, log_interval):
	tsne_path = path_to_save + "tsne/"
	Path(tsne_path).mkdir(parents=True, exist_ok=True)
	for i in range(1, iteration + 1):
		perplexity = np.random.rand() * 45 + 5
		early_exaggeration = np.random.rand() * 45 + 5
		learning_rate = np.random.rand() * 480 + 20
		emb = tsne_embedding(raw_data, perplexity, early_exaggeration, learning_rate)
		with open(tsne_path + str(i) + ".json", "w") as outfile:
			json.dump({
				"perplexity": perplexity,
				"early_exaggeration": early_exaggeration,
				"learning_rate": learning_rate,
				"emb": emb
			}, outfile)
		if i % log_interval == 0:
			print("t-SNE embedding generation: " + str(i) + "/" + str(iteration) + " finished")
	print("*** t-SNE embedding generation finished. ***")



def tsne_embedding(raw_data, perplexity, early_exaggeration, learning_rate):
	emb_data = TSNE(
		n_components=2, metric="euclidean", perplexity=perplexity, 
		early_exaggeration=early_exaggeration, learning_rate=learning_rate, random_state=1
	).fit_transform(raw_data)
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