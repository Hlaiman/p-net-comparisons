import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation


def get_model(inputs_num):

	inputs = Input(shape=(inputs_num))

	x = Dense(64, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(inputs)
	x = Activation('relu')(x)

	x = Dense(64, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(inputs)
	x = Activation('relu')(x)

	outputs = Dense(1, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')(x)

	return Model(inputs, outputs)