import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(os.getcwd() + '/../..')
sys.dont_write_bytecode = True

from tensorflow import keras
from utils.dataset_utils import prepare_images_from_directory, generate_multioutput_csv_from_dataset
from config import img_height, img_width, batch_size, classes_num, train_images_path, test_images_path, model_path, classes_num, with_data_normalization, color_mode

ds_train = prepare_images_from_directory(
	dir=train_images_path,
	image_size=(img_height, img_width),
	batch_size=batch_size,
	color_mode=color_mode,
	with_data_normalization = with_data_normalization
)

ds_test = prepare_images_from_directory(
	dir=test_images_path,
	image_size=(img_height, img_width),
	batch_size=batch_size,
	color_mode=color_mode,
	with_data_normalization = with_data_normalization
)

generate_multioutput_csv_from_dataset(ds_train, classes_num, "train_data.csv")
generate_multioutput_csv_from_dataset(ds_test, classes_num, "test_data.csv")