import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(os.getcwd() + '/..')
sys.dont_write_bytecode = True

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils
import pandas as pd
import numpy as np


def augment(x, y):
	image = tf.image.random_brightness(x, max_delta=0.05)
	image = tf.image.random_flip_left_right(image)
	return image, y


def normalize_image(x, y):
	image = tf.cast(x / 255., tf.float32)
	return image, y

def prepare_images_from_directory(dir, image_size, batch_size, color_mode, with_data_normalization = True):
	ds = tf.keras.preprocessing.image_dataset_from_directory(
		dir,
		labels="inferred",
		label_mode="int",  # categorical, binary
		color_mode=color_mode,
		batch_size=batch_size,
		image_size=image_size,  # reshape if not in this size
		shuffle=True,
	)

	if with_data_normalization:
		ds = ds.map(normalize_image)

	return ds


def prepare_image_generator_from_directory(dir, image_size, batch_size, color_mode, validation_split):
	datagen = ImageDataGenerator(rescale=1.0 / 255, data_format="channels_last", validation_split=validation_split)

	ds_train = datagen.flow_from_directory(
		dir,
		target_size=image_size,
		batch_size=batch_size,
		color_mode=color_mode,
		class_mode="sparse",
		shuffle=True,
		subset="training",
		seed=123,
	)

	ds_validation = datagen.flow_from_directory(
		dir,
		target_size=image_size,
		batch_size=batch_size,
		color_mode=color_mode,
		class_mode="sparse",
		shuffle=True,
		subset="validation",
		seed=123,
	)

	return ds_train, ds_validation


def generate_multioutput_csv_from_dataset(dataset, classes_num, name):
	full_dataset = []

	for x, y in dataset:
		y_array = [0] * classes_num
		y_array[y.numpy()[0]] = 100
		full_data = np.append(x.numpy()[0], y_array)
		full_dataset.append(full_data)

	columnsCount = len(full_dataset[0])
	columns = []
	for i in range(0, columnsCount):
		if i < columnsCount - classes_num:
			columns.append("in" + str(i+1))
		else:
			class_num = i - columnsCount + classes_num + 1
			columns.append("out" + str(class_num))

	df = pd.DataFrame(full_dataset, columns=columns)
	df.to_csv(name, index=False)
	print("Save %s" % name)


def get_classes_num(path):
	classes_num = 0
	for _, dirnames, _ in os.walk(path):
		classes_num += len(dirnames)

	return classes_num

def get_classes(path):
	classes = np.array(tf.io.gfile.listdir(str(path)))
	print('Classes:', classes)

	return classes

