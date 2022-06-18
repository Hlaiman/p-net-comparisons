import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation


def get_model(inputs_num, classes_num):

	inputs = Input(shape=(inputs_num))

	x = Dense(32, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(inputs)
	x = Activation('relu')(x)

	x = Dense(classes_num, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)
	outputs = Activation('softmax')(x)

	return Model(inputs, outputs)