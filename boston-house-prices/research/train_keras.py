import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_VISIBLE_DEVICES']='-1'

sys.dont_write_bytecode = True

sys.path.append(os.getcwd() + '/..')

import tensorflow as tf
import numpy as np
import sys
from tensorflow import keras
import matplotlib.pyplot as plt
from pprint import pprint
import json
import pandas as pd
from utils.dataset_utils import prepare_dataset_for_dense
from config import batch_size, keras_epochs, model_path, train_data_path
from model import get_model
import time
from datetime import datetime
import shutil

tf.compat.v1.reset_default_graph()

x_train, y_train, inputs_num = prepare_dataset_for_dense(train_data_path)

model = get_model(inputs_num)

model.summary()

model.compile(
	optimizer="adam",
    loss='mse',
    metrics=['mae', 'mse'],
)

time_start = time.perf_counter()

his = model.fit(
   x_train,
   y_train,
   epochs=keras_epochs,
   verbose=1,
   batch_size=batch_size,
   validation_split=0.0,
)

time_delta_dense = round((time.perf_counter() - time_start) * 1000)

model.save(model_path + "/keras_model.h5")

model_info = {"time": time_delta_dense}

with open(model_path + '/keras_info.json', 'w', encoding='utf-8') as f:
	json.dump(model_info, f, ensure_ascii=False, indent=4)



