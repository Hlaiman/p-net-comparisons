import os
import sys

wine_type = "white" #"white"

if wine_type == "red":
	data_path = "winequality-red.csv"
else:
	data_path = "winequality-white.csv"

train_data_path = 'train.csv'
test_data_path = 'test.csv'

model_path = "models"

batch_size = 1

keras_epochs = 2
pnet_epochs = 2

classes_num = 10
	
