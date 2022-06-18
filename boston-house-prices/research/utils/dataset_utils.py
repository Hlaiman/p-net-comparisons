import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(os.getcwd() + '/..')
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.keras import utils
import pandas as pd
import numpy as np


def generate_csv_from_dataset(dataset, train_data_path, test_data_path):

	train_data = []
	test_data = []

	for i in dataset.index:
		dataset_type = list(dataset.iloc[i])[0]

		if dataset_type == "train":
			train_data.append(list(dataset.iloc[i])[1:])
		else:
			test_data.append(list(dataset.iloc[i])[1:])


	df_train = pd.DataFrame(train_data, columns=None)
	df_train.to_csv(train_data_path, index=False)
	print("Save %s" % train_data_path)

	df_test = pd.DataFrame(test_data, columns=None)
	df_test.to_csv(test_data_path, index=False)
	print("Save %s" % test_data_path)

	print("Dataset created")

def normalize_dataset(path):
	df = pd.read_csv(path)
	df_without_y = df.iloc[:, :-1]
	mean = df_without_y.mean(axis=0)
	std = df_without_y.std(axis=0)

	df_without_y = (df_without_y - mean) / std
	normalized_df = df_without_y.assign(y=df.iloc[:, -1:])

	normalized_df.to_csv(path, index=False)

def prepare_dataset_for_dense(path):
	df = pd.read_csv(path)
	inputs_num = len(list(df.iloc[0])) - 1

	y_df = df.iloc[:, inputs_num:]
	x_df = df.iloc[:, :inputs_num]

	return x_df, y_df, inputs_num


