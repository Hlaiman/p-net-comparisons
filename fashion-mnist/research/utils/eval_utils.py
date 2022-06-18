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
from utils.dataset_utils import prepare_images_from_directory
from config import batch_size, img_height, img_width, model_path, classes_num, pnet_epochs, keras_epochs, color_mode


def test_keras_model(keras_model, images_path):

	ds = prepare_images_from_directory(
		dir=images_path,
		image_size=(img_height, img_width),
		color_mode=color_mode,
		batch_size=batch_size
	)

	current_example = 0
	right = 0
	inference_time = 0

	for x, y in ds:

		current_example += 1
		right_prediction = y.numpy()[0]

		time_start = time.perf_counter()
		result = keras_model(x)[0]
		inference_time += round((time.perf_counter() - time_start) * 1000000) / 1000
		result = np.argmax(result)

		if result == right_prediction:
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



def generate_pnet_training_history(log_path, data, classes):

	classes_num = len(classes)

	accuracy = []
	loss = []

	for filename in sorted(os.listdir(log_path), key=len):
		print(filename)
		df = pd.read_csv(log_path + "/" + filename, index_col=False)
		all = .0
		res = .0
		y_true_epoch = []
		y_pred_epoch = []
		for i in df.index:
			all += 1

			y_label = list(data.iloc[i])[-classes_num:]
			y_array = list(df.iloc[i])[-classes_num:]
			y_true_epoch.append(y_label)
			y_pred_epoch.append(y_array)
			y_pred = np.argmax(y_array)
			y_true = np.argmax(y_label)

			if (y_pred == y_true):
				res += 1

		print('Recognized', res, 'in', all, 'patterns from train dataset')
		accuracy_score = res / all

		cce = tf.keras.losses.CategoricalCrossentropy()
		loss_value = cce(y_true_epoch, y_pred_epoch).numpy()

		print("Accuracy: %s" % accuracy_score)
		print("Loss: %s" % loss_value)
		accuracy.append(accuracy_score)
		loss.append(loss_value)

	history = {
		"accuracy": accuracy,
		"loss": loss
	}

	return history


