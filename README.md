# dr-vae-backend

## How to run
```sh
cd model
python server.py
```

## Prepare model preset data 
**all files should follow name convention**
- latent_emb.json
    - [[0.0, 1], [1, -1], ... [0.0, -0.4]]
    - 2D array with shape ((#embeddings), 2)
    - for latent value exploration
    - UMAP result of latent_vector (5D to 2D)
- latent_vector.json
    - [[0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0, 0.0]... [-0.1, 0.2, 0.1, 0.2, 0.3]]
    - 2D array with shape ((#embeddings), (latent_dims))
- method_num.json 
    - [0, 0, 0, 1.....2]
    - 1D array with int(method_num) * #(embeddings)

```sh
# model.pt should be local
.
├── model_0
│   ├── num_points_0
│   │   ├── latent_emb.json
│   │   ├── latent_vector.json
│   │   ├── method_num.json
│   │   └── model.pt
│   └── num_points_1
│       ├── latent_emb.json
│       ├── latent_vector.json
│       ├── method_num.json
│       └── model.pt 
├── model_1
│   ├── num_points_0
│   │   ├── latent_emb.json
│   │   ├── latent_vector.json
...
```





