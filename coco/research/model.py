import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from pprint import pprint


def get_model(img_height, img_width, color_mode, classes_num):

	if color_mode == "rgb":
		channels = 3
	else:
		channels = 1

	inputs = Input(shape=(img_height, img_width, channels))

	x = Conv2D(16, 3, padding='same', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(inputs)
	x = Activation('relu')(x)
	x = MaxPooling2D(2, 2)(x)
	x = Dropout(0.2)(x)

	x = Conv2D(32, 3, padding='same', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(2, 2)(x)
	x = Dropout(0.2)(x)

	x = Conv2D(64, 3, padding='same', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D(2, 2)(x)
	x = Dropout(0.2)(x)

	x = Flatten()(x)

	x = Dense(classes_num, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
	outputs = Activation('softmax')(x)

	return Model(inputs, outputs)