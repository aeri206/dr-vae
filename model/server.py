

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
files = None
nn = None
nn_emb = None
nn_emb_NUM = 10
latent_dims = None
nNeighbors = 50
dataset = ''
pointNum = ''



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
    global files
    global nn
    global nn_emb
    global latent_dims
    global nNeighbors
    global dataset
    global pointNum


    pointNum = request.args.get("pointNum")
    dataset = request.args.get("dataset")
    idx = request.args.get("idx")
    print(idx)
    # print(path)
    tmp_path = './'+ dataset + '/' + pointNum + '/' + idx + '/'

    model_dir = './'+ dataset + '/' + pointNum + '/'
    
    vae = MODEL(tmp_path, int(pointNum))
    latent_dims = vae.latent_dims

    print(tmp_path +'latent_emb.json')
    with open(tmp_path +'latent_emb.json') as f:
        latent_emb = np.array(json.load(f))
        nNeighbors = max(10, min(int(len(latent_emb)* 0.1), 100))
        # print('nNeighbors', nNeighbors)
    with open(tmp_path +'method_num.json') as fi:
        latent_label = np.array(json.load(fi))
    with open(tmp_path +'latent_vector.json') as fil: # = point_count
        latent_vector = np.array(json.load(fil))
    with open(tmp_path + 'raw-filename.json') as file:
        files = np.array(json.load(file))

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
    
    return jsonify(vae.reconstruct(np.array(latent_values)).tolist()[0])
    
@app.route('/getlatentemb')
def get_latent_emb():
    global latent_emb
    global latent_label
    global latent_vector
    if latent_emb is None:
        emb = []
    else:
        emb = latent_emb.tolist()
    if latent_label is None:
        label = []
    else:
        label = latent_label.tolist()
    if latent_vector is None:
        vec = []
    else:
        vec = latent_vector.tolist()

    return jsonify({
        "emb": emb, 
        # "emb": latent_emb.tolist(), #에러 나옴
        "label": label,
        "vec": vec
    })

def read_file(path):
    with open(path, 'r') as f:
        res = json.load(f)
    return res


@app.route('/getknn')
def get_knn():
    global nn
    global vae
    global latent_label
    global files
    global latent_emb
    global latent_vector
    global dataset
    global pointNum
    n = int(request.args.get("n"))

    # ./mammoth/5000/2/latent_emb.json
    # './mammoth/5000/2/latent_embs.json'
    # './mammoth/5000/dr-result/umap-56.json'
    latent_values = getArrayData(request, "latentValues") #내가 요청보낸거
    knn = ((nn.kneighbors([latent_values])[1])[0]).tolist() #top indices
    # print(nn.kneighbors([latent_values])[0])#거리 : (1, 100) : (1, neghbors)
    # print(nn.kneighbors([latent_values])[1]) #indicies (1, neighbors)
    if (n > 0):
        knn = knn[:n]
        label_result = latent_label[knn]
        latent_result = latent_vector[knn]
        
        file_result = files[knn]
        data_dir = './'+ dataset + '/' + pointNum + '/dr-result/' 
        file_result = [read_file(data_dir + x) for x in file_result]
        
        
        x = [vae.reconstruct(x)[0].tolist() for x in latent_result]
        # print(vae.reconstruct(latent_result).shape) #(5, 2000, 2)
        return {
            "labels": label_result.tolist(),
            "embs": x,
            "latents": latent_result.tolist(),
            "files": file_result
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