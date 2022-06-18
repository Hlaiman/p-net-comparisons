import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(os.getcwd() + '/..')
sys.dont_write_bytecode = True

from pypnet import pypnet
from config import train_data_path, test_data_path, model_path, range
from utils.eval_utils import test_keras_model, test_pnet_model_with_csv
import tensorflow as tf
from tensorflow import keras
from pprint import pprint
import time
import json
from datetime import datetime

with open(model_path + '/keras_info.json', 'r', encoding='utf-8', errors='ignore') as json_data:
	 keras_training_time = json.load(json_data)["time"]

with open(model_path + '/pnet_info.json', 'r', encoding='utf-8', errors='ignore') as json_data:
     pnet_training_time = json.load(json_data)["time"]


# load Keras
keras_model = keras.models.load_model(model_path + "/keras_model.h5")

# load P-Net
pnet_model = pypnet.pnet()
pypnet.load(pnet_model, model_path + "/pnet_model.nnw")

# Test data
print("\n===== KERAS INFERENCE ======")
keras_right, \
keras_inference_time, \
keras_examples_count = test_keras_model(keras_model, test_data_path, range)

print("\n===== P-NET INFERENCE ======")
pnet_right, \
pnet_inference_time, \
pnet_examples_count = test_pnet_model_with_csv(pnet_model, test_data_path, range)

# Results
print("\n===== COMPARISON RESULTS ======")

keras_speed = keras_inference_time/keras_examples_count
pnet_speed = pnet_inference_time/pnet_examples_count

print("\n===== Speed:")
print("\nKERAS average speed is {} ms".format(round(keras_speed, 4)))
print("P-NET average speed is {} ms".format(round(pnet_speed, 4)))

print("\nP-NET predicts {} times faster than KERAS".format(round(keras_speed / pnet_speed, 2)))

keras_accuracy = keras_right/keras_examples_count
pnet_accuracy = pnet_right/pnet_examples_count

print("\n===== Accuracy:")
print("\nKERAS accuracy is {}".format(round(keras_accuracy, 4)))
print("P-NET accuracy is {}".format(round(pnet_accuracy, 4)))

print("\nP-NET is {} times more accurate than KERAS".format(round(pnet_accuracy / keras_accuracy, 2)))

print("\n===== Training time:")
print("\nKERAS training time: %s ms" % keras_training_time)
print("P-NET training time: %s ms" % pnet_training_time)

print("\nP-NET trains {} times faster than KERAS".format(round(keras_training_time / pnet_training_time, 2)))



