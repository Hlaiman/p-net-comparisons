import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(os.getcwd() + '/..')
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.keras import utils
import pandas as pd
import numpy as np


def generate_multioutput_csv_from_dataset(dataset, classes_num, train_data_path, test_data_path):

	train_data = []
	test_data = []

	for i in dataset.index:
		dataset_type = list(dataset.iloc[i])[0]

		y_array = [0] * classes_num
		y_array[list(dataset.iloc[i])[-1:][0]] = 100

		if dataset_type == "train":
			train_data.append(list(dataset.iloc[i])[1:-1] + y_array)
		else:
			test_data.append(list(dataset.iloc[i])[1:-1] + y_array)


	df_train = pd.DataFrame(train_data, columns=None)
	df_train.to_csv(train_data_path, index=False)
	print("Save %s" % train_data_path)

	df_test = pd.DataFrame(test_data, columns=None)
	df_test.to_csv(test_data_path, index=False)
	print("Save %s" % test_data_path)

	print("Dataset created")

def normalize_dataset(path, classes_num):
	df = pd.read_csv(path)
	df_without_y = df.iloc[:, :-classes_num]
	mean = df_without_y.mean(axis=0)
	std = df_without_y.std(axis=0)

	df_without_y = (df_without_y - mean) / std
	normalized_df = pd.concat([df_without_y, df.iloc[:, -classes_num:]], axis=1)

	normalized_df.to_csv(path, index=False)

def prepare_dataset_for_dense(data_path, classes_num):
	df_r = pd.read_csv(data_path)
	inputs_num = len(list(df_r.iloc[0])) - classes_num
	df = df_r.sample(frac=1)
	y_df = df.iloc[:, inputs_num:]
	x_df = df.iloc[:, :inputs_num]

	y_number = []
	for i in range(len(y_df)):
		y_array = list(y_df.iloc[i])
		y_number.append(np.argmax(y_array))

	y = utils.to_categorical(y_number, classes_num)

	return x_df, y, inputs_num


