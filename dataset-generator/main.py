## 실행방법
## pip으로 필요한거를 다 깔거나
## 다음 명령어를 통해 conda 환경에 진입해서 사용
## - conda env create --file enviroment.yaml
## - conda activate dr-vae-dataset

## Parameter 설명
## -d : 데이터셋 이름 
## -s : 데이터셋이 가지고 있는 샘플 (포인트 개수)
#### 데이터셋 이름이 dataAAA고 샘플 수가 5000일 때 데이터는 dataset/dataAAA/5000/ 디렉토리에 저장되어야 함
#### raw.json, label.json 필요 (각각 원본 데이터 / 레이블 정보를 담고 있는 arry 형태)
## -n : 최종적으로 만들어질 embedding의 개수
## -p : 임베딩들이 저장될 경로 ("/"로 끝나야 함)
## -l : 임베딩들을 생성할 때 이 개수만큼의 임베딩을 생성할 때마다 log을 찍어 줌
## -c : DR 메소드 하나당 임베딩을 만들기 위해 할당되는 cpu 수 (20~25가 적당한듯 다른 사람이 아무도 안쓸떄는 림으면 개느림)
#### n이 5*c의 배수여야 함 (그래야 딱 떨어지게 나옴)

## 어떤 dataset을 직접 적용해보고 싶으면 맞춰서 json으로 raw, label 파일 다른데서 생성해서 형식 맞춰 디렉토리 만들고 넣어주면 됨

from pathlib import Path
from multiprocessing import Process, cpu_count
from embedding import *

import argparse
import json
import numpy as np
import time

parser = argparse.ArgumentParser(description="Dataset Generation for DR_VAE", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-d', '--dataset', help="The dataset to generate DR embedding -- name of directory")
parser.add_argument('-s', '--size',  help="The size of dataset -- name of subdirectory\n" + " (under the directory denoting --dataset)")
parser.add_argument('-n', '--num', help="The total number of embeddings to be created")
parser.add_argument('-p', '--path', default="./result/", help="Designate directory for saving generated embeddings")
parser.add_argument('-l', '--log-interval', default=50, help="print current status per log interver")
parser.add_argument('-c', '--cpu-per-method', default=2, help="the number of allocated cpu per DR method")


args = parser.parse_args()

dataset       = args.dataset
size          = args.size
num           = int(args.num)
path          = args.path

log_interval    = int(args.log_interval)
cpu_per_method  = int(args.cpu_per_method)


### Read dataset / label
path_to_dataset = "./dataset/" + dataset + "/" + size + "/"

with open(path_to_dataset + "raw.json") as raw_file:
	raw = np.array(json.load(raw_file))
with open(path_to_dataset + "label.json") as label_file:
	label = np.array(json.load(label_file))




## Path to save file
path_to_save = path + "/" + dataset + "/" + size + "/"
Path(path_to_save).mkdir(parents=True, exist_ok=True)

## Save raw / label data
with open(path_to_save + "raw.json", "w") as raw_save_file:
	json.dump(raw.tolist(), raw_save_file)
with open(path_to_save + "label.json", "w") as label_save_file:
	json.dump(label.tolist(), label_save_file)



target_list = [
	umap_generate_embeddings,
	tsne_generate_embeddings,
	isomap_generate_embeddings,
	lle_generate_embeddings,
	densmap_generate_embeddings
]

iteration = (num // len(target_list)) // cpu_per_method



print("*** Embedding Generation Start!! ***")
print("*** CPU #:", cpu_count(), "***")


start_time = time.time()

process_list = []
for target in target_list:
	for i in range(1, cpu_per_method + 1):
		process = Process(target=target, args=(raw, path_to_save, iteration, log_interval, i))
		process_list.append(process)
		process.start()
	for i in range(cpu_per_method):
		process.join()


end_time = time.time()


print("*** Generation Finished. ***")
print("Elapsed time:", round(end_time - start_time, 3), "seconds" )









