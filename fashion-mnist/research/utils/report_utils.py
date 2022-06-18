import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(os.getcwd() + '/../..')
sys.dont_write_bytecode = True

from pypnet import pypnet
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import visualkeras

a4_landscape_size = [11.7, 8.3]

def plot_graph(data, name, legend):
	  fig = plt.figure()
	  for paramenter, parameter_name in data:
		  plt.plot(paramenter, label=parameter_name)

	  plt.title(name)
	  plt.ylabel(legend[0])
	  plt.xlabel(legend[1])
	  plt.legend(loc="upper left")
	  fig.set_size_inches(a4_landscape_size)
	  return fig

def set_image(image):
	  fig = plt.figure()
	  plt.axis('off')
	  plt.title('Model architecture')
	  plt.imshow(image)

	  fig.set_size_inches(a4_landscape_size)
	  return fig

def get_title_page(title):
	fig = plt.figure()
	plt.axis('off')
	plt.text(0.5, 0.6, 'Report', ha='center', va='center', fontsize=40)
	plt.text(0.5, 0.4, title, ha='center', va='center', fontsize=22)
	fig.set_size_inches(a4_landscape_size)

	return fig


def plot_confusion_matrix(y_true, y_pred, title, classes):
	fig = plt.figure()
	confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
	sns.heatmap(confusion_mtx, xticklabels=classes, yticklabels=classes, annot=True, fmt='g')
	plt.title(title + ' confusion matrix')
	plt.xticks(rotation=45)
	plt.yticks(rotation=45)
	plt.xlabel('Prediction')
	plt.ylabel('Label')
	fig.subplots_adjust(bottom=0.2, left=0.2)
	fig.set_size_inches(a4_landscape_size)

	return fig

def generate_y_with_dense(model, ds):
	samples = []
	labels = []

	for sample, label in ds:
		samples.append(sample.numpy()[0])
		labels.append(label.numpy()[0])

	samples = np.array(samples)
	labels = np.array(labels)

	y_pred = np.argmax(model.predict(samples), axis=1)
	y_true = labels

	return y_true, y_pred

def generate_y_with_pnet_multioutput(partial_model, pnet, ds):
	labels = []
	y_pred = []
	for sample, label in ds:
		labels.append(label.numpy()[0])
		flatten_result = partial_model(sample)
		result = pypnet.compute(pnet, list(flatten_result.numpy()[0]))
		y_pred.append(np.argmax(result))

	y_pred = np.array(y_pred)
	labels = np.array(labels)
	y_true = labels

	return y_true, y_pred


def generate_y_with_pnet_multioutput_from_csv(pnet, ds, classes_num):
	labels = []
	y_pred = []
	for i in ds.index:
		label_onehot = list(ds.iloc[i])[-classes_num:]
		label = np.argmax(label_onehot)
		labels.append(label)
		sample = list(ds.iloc[i])[:-classes_num]
		result = pypnet.compute(pnet, sample)
		result = np.argmax(result)
		y_pred.append(result)

	y_pred = np.array(y_pred)
	labels = np.array(labels)
	y_true = labels

	return y_true, y_pred

def get_model_summary(model):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	stringlist = []
	model.summary(line_length=100, print_fn=lambda x: stringlist.append(x))
	short_model_summary = "\n".join(stringlist)

	plt.text(0.5, 0.5, short_model_summary, {'fontsize': 7}, ha='center', va='center', transform=ax.transAxes, fontproperties = 'monospace')
	plt.axis('off')
	fig.set_size_inches(a4_landscape_size)

	return fig

def save_report_keras(model, datasets, classes, history, name, save_path):

	report_dist_path = "../reports/" + save_path

	title_page = get_title_page(name)

	model_summary = get_model_summary(model)

	loss = plot_graph(
		[[history['loss'], 'Training data']], # [history['val_loss'], 'Validation data']
		name="Training losses",
		legend=['Loss', 'No. epoch']
	)

	accuracy = plot_graph(
		[[history['accuracy'], 'Training data']], # [history['val_accuracy'], 'Validation data']
		name="Model accuracy",
		legend=['Accuracy', 'No. epoch']
	)

	if not os.path.exists(report_dist_path):
		os.mkdir(report_dist_path)

	pp = PdfPages(report_dist_path + "/" + name + '.pdf')
	pp.savefig(title_page)
	pp.savefig(model_summary)

	model_image = set_image(visualkeras.layered_view(
		model,
		legend=True,
		scale_xy=2,
		scale_z=1,
		draw_funnel=False,
		min_z = 10,
		min_xy = 10,
		max_z = 200,
		max_xy = 1000
	))
	pp.savefig(model_image)

	pp.savefig(loss)
	pp.savefig(accuracy)

	for ds, title in datasets:
		y_true, y_pred = generate_y_with_dense(model, ds)
		confusion_matrix = plot_confusion_matrix(y_true, y_pred, title = title, classes = classes)
		pp.savefig(confusion_matrix)

	pp.close()


def save_report_pnet(datasets, classes, history, save_path, name, pnet):

	report_dist_path = "../reports/" + save_path

	title_page = get_title_page(name)

	loss = plot_graph(
		[[history['loss'], 'Training data']],
		name="Training losses",
		legend=['Loss', 'No. epoch']
	)

	accuracy = plot_graph(
		[[history['accuracy'], 'Training data']],
		name="Model accuracy",
		legend=['Accuracy', 'No. epoch']
	)

	if not os.path.exists(report_dist_path):
		os.mkdir(report_dist_path)

	pp = PdfPages(report_dist_path + "/" + name + '.pdf')
	pp.savefig(title_page)
	pp.savefig(loss)
	pp.savefig(accuracy)

	for ds, title in datasets:
		y_true, y_pred = generate_y_with_pnet_multioutput_from_csv(pnet, ds, classes_num = len(classes))
		confusion_matrix = plot_confusion_matrix(y_true, y_pred, title = title, classes = classes)
		pp.savefig(confusion_matrix)

	pp.close()
