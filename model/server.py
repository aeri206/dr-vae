




# def call():
#     v = MODEL(latent_dims)
#     vec = np.random.rand(latent_dims)
#     start_time = time.time()
#     v.reconstruct(vec)
#     print("--- %s seconds ---" % (time.time() - start_time))

# call()


from flask import Flask, request, jsonify
from flask_cors import CORS

from model import MODEL
import numpy as np
import time
import json

from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app)

latent_dims = 5
vae = None
latent_emb = None
latent_label = None
latent_vector = None
nn = None



def getArrayData(request, key_name):
    array_data = request.args.get(key_name)
    array_data = np.array(json.loads(array_data)["data"]).astype(np.float32)
    return array_data

@app.route('/reconstruction')
def reconstruction():
    global vae
    latent_values = getArrayData(request, "latentValues")
    return jsonify(vae.reconstruct(np.array(latent_values)).tolist()[0])
    
@app.route('/getlatentemb')
def get_latent_emb():
    global latent_dims
    global latent_label
    return jsonify({
        "emb": latent_emb.tolist(),
        "label": latent_label.tolist()
    })

@app.route('/getknn')
def get_knn():
    global nn
    global latent_label
    global latent_emb
    latent_values = getArrayData(request, "latentValues")
    # print(latent_values)
    knn = ((nn.kneighbors([latent_values])[1])[0]).tolist()
    label_result = latent_label[knn]
    coordinate = np.sum(latent_emb[knn], axis=0) / 100
    return {
        "labels": label_result.tolist(),
        "coor": coordinate.tolist()
    }
    # return "succcess"


if __name__ == '__main__':
    vae = MODEL(latent_dims)
    with open('../latent_emb.json') as f:
        latent_emb = np.array(json.load(f))
    with open('../method_num.json') as fi:
        latent_label = np.array(json.load(fi))
    with open('../latent_vector.json') as fil:
        latent_vector = json.load(fil)
    nn = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(latent_vector)
    app.run(debug=True)
    