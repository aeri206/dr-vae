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
def umap_generate_embeddings(raw_data, path_to_save, iteration, log_interval, set_num):
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
			print("UMAP embedding set #" + str(set_num) + " generation: " + str(i) + "/" + str(iteration) + " finished")
	print("*** UMAP embedding set #" + str(set_num) + " generation finished. ***")


def umap_embedding(raw_data, min_dist, n_neighbors):
	umap_instance = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
	emb_data = umap_instance.fit_transform(raw_data)
	emb_data = emb_data.tolist()
	return emb_data

'''
TSNE
'''
def tsne_generate_embeddings(raw_data, path_to_save, iteration, log_interval, set_num):
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
			print("t-SNE embedding set #" + str(set_num) + " generation: " + str(i) + "/" + str(iteration) + " finished")
	print("*** t-SNE embedding set #" + str(set_num) + " generation finished. ***")


def tsne_embedding(raw_data, perplexity, early_exaggeration, learning_rate):
	emb_data = TSNE(
		n_components=2, metric="euclidean", perplexity=perplexity, 
		early_exaggeration=early_exaggeration, learning_rate=learning_rate, random_state=1
	).fit_transform(raw_data)
	emb_data = emb_data.tolist()
	return emb_data

'''
ISOMAP
'''

def isomap_generate_embeddings(raw_data, path_to_save, iteration, log_interval, set_num):
	isomap_path = path_to_save + "isomap/"
	Path(isomap_path).mkdir(parents=True, exist_ok=True)
	for i in range(1, iteration + 1):
		n_neighbors = np.random.randint(3, 101)
		emb = isomap_embedding(raw_data, n_neighbors)
		with open(isomap_path + str(i) + ".json", "w") as outfile:
			json.dump({
				"n_neighbors": n_neighbors,
				"emb": emb
			}, outfile)
		if i % log_interval == 0:
			print("ISOMAP embedding set #" + str(set_num) + " generation: " + str(i) + "/" + str(iteration) + " finished")
	print("*** ISOMAP embedding set #" + str(set_num) + " generation finished. ***")

def isomap_embedding(raw_data, n_neighbors):
	emb_data = Isomap(n_neighbors=n_neighbors).fit_transform(raw_data)
	emb_data = emb_data.tolist()
	return emb_data


'''
LLE
'''

def lle_generate_embeddings(raw_data, path_to_save, iteration, log_interval, set_num):
	lle_path = path_to_save + "lle/"
	Path(lle_path).mkdir(parents=True, exist_ok=True)
	for i in range(1, iteration + 1):
		n_neighbors = np.random.randint(3, 101)
		emb = lle_embedding(raw_data, n_neighbors)
		with open(lle_path + str(i) + ".json", "w") as outfile:
			json.dump({
				"n_neighbors": n_neighbors,
				"emb": emb
			}, outfile)
		if i % log_interval == 0:
			print("LLE embedding set #" + str(set_num) + " generation: " + str(i) + "/" + str(iteration) + " finished")
	print("*** LLE embedding set #" + str(set_num) + " generation finished. ***")	


def lle_embedding(raw_data, n_neighbors):
	lle = LocallyLinearEmbedding(n_neighbors=n_neighbors)
	emb_data = lle.fit_transform(raw_data)
	return emb_data.tolist()

'''
DENSMAP
'''

def densmap_generate_embeddings(raw_data, path_to_save, iteration, log_interval, set_num):
	densmap_path = path_to_save + "densmap/"
	Path(densmap_path).mkdir(parents=True, exist_ok=True)
	for i in range(1, iteration + 1):
		min_dist = np.random.rand()
		n_neighbors = np.random.randint(3, 101)
		emb = densmap_embedding(raw_data, min_dist, n_neighbors)
		with open(densmap_path + str(i + (set_num - 1) * iteration) + ".json", "w") as outfile:
			json.dump({
				"min_dist" : min_dist,
				"n_neighbors" : n_neighbors,
				"emb": emb
			}, outfile)
		if i % log_interval == 0:
			print("DENSMAP embedding set #" + str(set_num) + " generation: " + str(i) + "/" + str(iteration) + " finished")
	print("*** DENSMAP embedding set #" + str(set_num) + " generation finished. ***")

def densmap_embedding(raw_data, min_dist, n_neighbors):
	umap_instance = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, densmap=True)
	emb_data = umap_instance.fit_transform(raw_data)
	emb_data = emb_data.tolist()
	return emb_data