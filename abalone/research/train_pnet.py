import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.dont_write_bytecode = True
sys.path.append(os.getcwd() + '/../..')

from pypnet import pypnet

from config import pnet_epochs, batch_size, model_path, train_data_path
import time
import json
from pprint import pprint
import pandas as pd
import cv2
import numpy as np
import shutil


df = pd.read_csv(train_data_path)
df = df.sample(frac=1)

inputs_num = len(list(df.iloc[0])) - 1

print("Inputs count: %s" % inputs_num)

pnet = pypnet.new(
	inputs=inputs_num,
	epoch=pnet_epochs,
	outputs=1,
	autointervals=True,
	#intervals=10,
	density=1,
)

pypnet.load(pnet, train_data_path)

print(pypnet.info(pnet))

time_start = time.perf_counter()

print(pypnet.train(pnet))

time_delta_pnet = round((time.perf_counter() - time_start) * 1000)

model_info = {"time": time_delta_pnet}

with open(model_path + '/pnet_info.json', 'w', encoding='utf-8') as f:
	json.dump(model_info, f, ensure_ascii=False, indent=4)

if not os.path.exists(model_path):
	os.mkdir(model_path)

print('Save P-Net .nnw configuration to', pypnet.save(pnet, 'pnet_model.nnw'))
os.replace("pnet_model.nnw", model_path + "/pnet_model.nnw")
