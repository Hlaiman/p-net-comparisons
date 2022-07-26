import os
import sys

sys.dont_write_bytecode = True
sys.path.append(os.getcwd() + '/../..')

from pypnet import pypnet

from config import img_height, img_width, classes_num, pnet_epochs, batch_size, model_path, classes_num, train_images_path, with_pnet_report
from utils.dataset_utils import get_classes
from utils.report_utils import save_report_pnet
from utils.eval_utils import generate_pnet_training_history
import time
import json
from pprint import pprint
import pandas as pd
import cv2
import numpy as np
import shutil

df = pd.read_csv('train_data.csv')
df = df.sample(frac=1)

inputs_num = len(list(df.iloc[0])) - classes_num

print("Inputs count: %s" % inputs_num)

pnet = pypnet.new(
	layers=2,
	inputs=inputs_num,
	epoch=pnet_epochs,
	outputs=classes_num,
	intervals=12,
	autointervals=True
)

pypnet.load(pnet, 'train_data.csv')

print(pypnet.info(pnet))

if with_pnet_report and os.path.exists('log'):
	shutil.rmtree('log')

time_start = time.perf_counter()

if with_pnet_report:
	print(pypnet.train(pnet, 'log\pnet.csv'))
else:
	print(pypnet.train(pnet))

time_delta_pnet = round((time.perf_counter() - time_start) * 1000)

model_info = {"time": time_delta_pnet}

with open(model_path + '/pnet_info.json', 'w', encoding='utf-8') as f:
	json.dump(model_info, f, ensure_ascii=False, indent=4)

if not os.path.exists(model_path):
	os.mkdir(model_path)

print('Save P-Net .nnw configuration to', pypnet.save(pnet, 'pnet_model.nnw'))
os.replace("pnet_model.nnw", model_path + "/pnet_model.nnw")

if with_pnet_report:
	# Report
	classes = get_classes(train_images_path)
	train_data = pd.read_csv('train_data.csv', index_col = False)
	test_data = pd.read_csv('test_data.csv', index_col = False)
	history = generate_pnet_training_history('log', train_data, classes)

	if os.path.exists('log'):
		shutil.rmtree('log')

	save_report_pnet(
		datasets = [[train_data, "Train data"], [test_data, "Test data"]],
		classes = classes,
		history = history,
		save_path = "p-net",
		name = "P-Net",
		pnet = pnet,
	)
