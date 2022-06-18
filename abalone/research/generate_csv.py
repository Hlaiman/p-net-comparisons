import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_VISIBLE_DEVICES']='-1'

sys.path.append(os.getcwd() + '/../..')
sys.dont_write_bytecode = True

from tensorflow import keras
from utils.dataset_utils import generate_csv_from_dataset, normalize_dataset
import pandas as pd
from config import train_data_path, test_data_path, data_path

df = pd.read_csv(data_path, sep=";")

generate_csv_from_dataset(df, train_data_path, test_data_path)

normalize_dataset(train_data_path)
normalize_dataset(test_data_path)

