




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


app = Flask(__name__)
CORS(app)

latent_dims = 5
vae = None
latent_emb = None
latent_label = None



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
        "emb": latent_emb,
        "label": latent_label
    })



if __name__ == '__main__':
    vae = MODEL(latent_dims)
    with open('../latent_emb.json') as f:
        latent_emb = json.load(f)
    with open('../method_num.json') as fi:
        latent_label = json.load(fi)
    app.run(debug=True)
    