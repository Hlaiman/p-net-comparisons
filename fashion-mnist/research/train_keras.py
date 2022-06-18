import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sys.dont_write_bytecode = True

sys.path.append(os.getcwd() + '/..')
from pypnet import pypnet

import tensorflow as tf
import numpy as np
import sys
from tensorflow import keras
import matplotlib.pyplot as plt
from pprint import pprint
import json
import pandas as pd
from utils.dataset_utils import prepare_image_generator_from_directory, prepare_images_from_directory
from utils.report_utils import save_report_keras
from config import img_height, img_width, batch_size, classes_num, keras_epochs, model_path, train_images_path, test_images_path, with_keras_report, color_mode
from model import get_model
import time
from datetime import datetime
import shutil

tf.compat.v1.reset_default_graph()

ds_train, ds_validation = prepare_image_generator_from_directory(
	dir=train_images_path,
	image_size=(img_height, img_width),
	batch_size=batch_size,
	color_mode=color_mode,
	validation_split=0.0
)

model = get_model(
	img_height,
	img_width,
	color_mode=color_mode,
	classes_num=classes_num
)

print(model.summary())

model.compile(
	optimizer=keras.optimizers.Adam(),
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=["accuracy"],
)

time_start = time.perf_counter()

his = model.fit(
	ds_train,
	epochs=keras_epochs,
	verbose=1,
	batch_size=batch_size,
	validation_data=ds_validation,
)

time_delta_dense = round((time.perf_counter() - time_start) * 1000)

model.save(model_path + "/keras_model.h5")

model_info = {"time": time_delta_dense}

with open(model_path + '/keras_info.json', 'w', encoding='utf-8') as f:
	json.dump(model_info, f, ensure_ascii=False, indent=4)

if with_keras_report:
	# Report
	classes = list(ds_train.class_indices)

	ds_test = prepare_images_from_directory(
		dir=test_images_path,
		image_size=(img_height, img_width),
		batch_size=batch_size,
		color_mode=color_mode
	)

	ds_train = prepare_images_from_directory(
		dir=train_images_path,
		image_size=(img_height, img_width),
		batch_size=batch_size,
		color_mode=color_mode
	)

	save_report_keras(
		model = model,
		datasets = [[ds_test, 'Test data'], [ds_train, 'Train data']],
		classes = classes,
		history = his.history,
		save_path = "keras",
		name = "Keras",
	)


