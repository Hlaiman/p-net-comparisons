import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(os.getcwd() + '/..')
sys.dont_write_bytecode = True

from pypnet import pypnet
import tensorflow as tf
from tensorflow import keras
import time
import pandas as pd
import cv2
import numpy as np


def test_keras_model(keras_model, test_data_path, classes_num):

	df = pd.read_csv(test_data_path)

	current_example = 0
	right = 0
	inference_time = 0

	inputs_num = len(list(df.iloc[0])) - classes_num

	y_df = df.iloc[:, inputs_num:]
	x_df = df.iloc[:, :inputs_num]

	for i in x_df.index:

		current_example += 1

		test_sample = tf.constant([x_df.iloc[i]])

		time_start = time.perf_counter()
		result = keras_model(test_sample)[0]
		inference_time += round((time.perf_counter() - time_start) * 1000000) / 1000

		result = np.argmax(result)

		y_array = list(df.iloc[i])[-classes_num:]
		y = np.argmax(y_array)

		if result == y:
			right += 1


	return right, inference_time, current_example

def test_multioutput_pnet_model_with_csv(pnet, test_data_path, classes_num):
	df = pd.read_csv(test_data_path)

	current_example = 0
	right = 0
	inference_time = 0

	for i in df.index:
		current_example += 1
		test_sample = list(df.iloc[i])[:-classes_num]

		time_start = time.perf_counter()
		result = pypnet.compute(pnet, test_sample)
		inference_time += round((time.perf_counter() - time_start) * 1000000) / 1000
		result = np.argmax(result)

		y_array = list(df.iloc[i])[-classes_num:]
		y = np.argmax(y_array)

		if (result == y):
			right += 1

	return right, inference_time, current_example



