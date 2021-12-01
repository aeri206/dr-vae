

from glob import glob
from flask import Flask, request, jsonify
from flask_cors import CORS

from model import MODEL
import numpy as np
import time
import json

from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app)

vae = None
latent_emb = None
latent_label = None #method_num
latent_vector = None
nn = None
nn_emb = None
nn_emb_NUM = 10
latent_dims = None
nNeighbors = 50



def getArrayData(request, key_name):
    array_data = request.args.get(key_name)
    array_data = np.array(json.loads(array_data)["data"]).astype(np.float32)
    return array_data

@app.route('/reload')
def reload():
    global vae
    global latent_emb
    global latent_label
    global latent_vector
    global nn
    global nn_emb
    global latent_dims
    global nNeighbors
    dataset = request.args.get("dataset")
    pointNum = request.args.get("pointNum")
    # print(path)
    model_dir = './'+ dataset + '/' + pointNum + '/'
    model_path = model_dir + 'model.pt'
    vae = MODEL(model_path, dataset, int(pointNum))
    latent_dims = vae.latent_dims


    with open(model_dir +'latent_emb.json') as f:
        latent_emb = np.array(json.load(f))
        nNeighbors = max(10, min(int(len(latent_emb)* 0.1), 100))
        print('nNeighbors', nNeighbors)
    with open(model_dir +'method_num.json') as fi:
        latent_label = np.array(json.load(fi))
    with open(model_dir +'latent_vector.json') as fil: # = point_count
        latent_vector = np.array(json.load(fil))
    nn = NearestNeighbors(n_neighbors=nNeighbors, algorithm='ball_tree').fit(latent_vector)
    #latent vector : #embedding * latent_dim
    nn_emb = NearestNeighbors(n_neighbors=nn_emb_NUM, algorithm='ball_tree').fit(latent_emb)
    #latent_emb : #embdddning * UMAP result
    return jsonify(vae.latent_dims)

@app.route('/getdims')
def get_latent_dims():
    global latent_dims
    if latent_dims is None:
        latent_dims = reload()
        return latent_dims
    else:
        return jsonify(latent_dims)


@app.route('/reconstruction')
def reconstruction():
    global vae
    latent_values = getArrayData(request, "latentValues")
    #return shape : (point_count, 2)
    # print(latent_values)

    return jsonify(vae.reconstruct(np.array(latent_values)).tolist()[0])
    
@app.route('/getlatentemb')
def get_latent_emb():
    global latent_emb
    global latent_label
    return jsonify({
        "emb": latent_emb.tolist(),
        "label": latent_label.tolist()
    })

@app.route('/getknn')
def get_knn():
    global nn
    global vae
    global latent_label
    global latent_emb
    global latent_vector
    n = int(request.args.get("n"))
    latent_values = getArrayData(request, "latentValues") #내가 요청보낸거
    knn = ((nn.kneighbors([latent_values])[1])[0]).tolist() #top indices
    # print(nn.kneighbors([latent_values])[0])#거리 : (1, 100) : (1, neghbors)
    # print(nn.kneighbors([latent_values])[1]) #indicies (1, neighbors)
    if (n > 0):
        knn = knn[:n]
        label_result = latent_label[knn]
        latent_result = latent_vector[knn]
        
        x = [vae.reconstruct(x)[0].tolist() for x in latent_result]
        # print(vae.reconstruct(latent_result).shape) #(5, 2000, 2)
        return {
            "labels": label_result.tolist(),
            "embs": x,
            "latents": latent_result.tolist()
        }
    else:
        label_result = latent_label[knn] #class? (neighbors, )
        coordinate = np.sum(latent_emb[knn], axis=0) / nNeighbors
        return {
            "labels": label_result.tolist(),
            "coor": coordinate.tolist()
        }

@app.route('/latentcoortoothers')
def latent_coor_to_others():
    global nn_emb
    global latent_vector
    global vae

    latent_coor = getArrayData(request, "coor")
    
    knn = ((nn_emb.kneighbors([latent_coor])[1])[0]).tolist()
    label_result = latent_label[knn]
    vector = np.sum(latent_vector[knn], axis=0) / nn_emb_NUM
    
    return {
        "labels": label_result.tolist(),
        "latent_values": vector.tolist(),
        "emb": vae.reconstruct(vector).tolist()[0]

    }

if __name__ == '__main__':
    app.run(debug=True)