import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(os.getcwd() + '/../..')
sys.dont_write_bytecode = True

from tensorflow import keras
from utils.dataset_utils import generate_multioutput_csv_from_dataset, normalize_dataset
import pandas as pd
from config import train_data_path, test_data_path, data_path, classes_num

df = pd.read_csv(data_path, sep=";")

generate_multioutput_csv_from_dataset(df, classes_num, train_data_path, test_data_path)

normalize_dataset(train_data_path, classes_num)
normalize_dataset(test_data_path, classes_num)