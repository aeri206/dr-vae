## Parameter 설명
## -d : 데이터셋 이름 
## -n : 데이터셋이 가지고 있는 샘플 (포인트 개수)
#### 데이터셋 이름이 dataAAA고 샘플 수가 5000일 때 데이터는 dataset/dataAAA/5000/ 디렉토리에 저장되어야 함
#### raw.json, label.json 필요 (각각 원본 데이터 / 레이블 정보를 담고 있는 arry 형태)
## -s : 최종적으로 만들어질 embedding의 개수
## -p : 임베딩들이 저장될 경로 ("/"로 끝나야 함)

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
from embedding import *

import argparse
import json
import numpy as np

parser = argparse.ArgumentParser(description="Dataset Generation for DR_VAE", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-d', '--dataset', help="The dataset to generate DR embedding -- name of directory")
parser.add_argument('-s', '--size',  help="The size of dataset -- name of subdirectory\n" + " (under the directory denoting --dataset)")
parser.add_argument('-n', '--num', help="The total number of embeddings to be created")
parser.add_argument('-p', '--path', default="./result/", help="Designate directory for saving generated embeddings")


args = parser.parse_args()

dataset = args.dataset
size    = args.size
num     = int(args.num)
path    = args.path


### Read dataset / label
path_to_dataset = "./dataset/" + dataset + "/" + size + "/"

with open(path_to_dataset + "raw.json") as raw_file:
	raw = np.array(json.load(raw_file))
with open(path_to_dataset + "label.json") as label_file:
	label = np.array(json.load(label_file))



raw = raw[:100]
label = label[:100]

## Path to save file
path_to_save = path + "/" + dataset + "/" + size + "/"
Path(path_to_save).mkdir(parents=True, exist_ok=True)

## Save raw / label data
with open(path_to_save + "raw.json", "w") as raw_save_file:
	json.dump(raw.tolist(), raw_save_file)
with open(path_to_save + "label.json", "w") as label_save_file:
	json.dump(label.tolist(), label_save_file)

## Save umap data
umap_generate_embeddings(raw, path_to_save, 100)









