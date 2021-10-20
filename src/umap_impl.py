import torch
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import umap.umap_ as umap
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import time
import csv
import matplotlib.cm as cm
import os

import dataloader_umap

mean = (0.3499, 0.4374, 0.5199)
std = (0.1623, 0.1894, 0.1775)


classes = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon', 'truck']


def get_correct_list(srcFile):
	src_file = open(srcFile, "r")
	csv_file = csv.DictReader(src_file)
	csv_file_data = list(csv_file)
	bool_list = []
	for i in range(len(csv_file_data)):
		row = csv_file_data[i]
		if int(row['Predicted Label']) == int(row['True Label']):
			bool_list.append(True)
		else:
			bool_list.append(False)
	return bool_list


def get_first_object(testFile):
	test_file_csv = csv.DictReader(open(testFile, "r"))
	test_file_csv_data = list(test_file_csv)
	first_obj = {}
	for i in range(len(test_file_csv_data)):
		row = test_file_csv_data[i]
		cl_id = int(row['Class ID'])
		obj = int(row['Object'])
		if cl_id not in first_obj.keys():
			first_obj[cl_id] = obj
	# print(first_obj)
	return first_obj


def get_selected_ims(testFile, obj_dict):
	test_file_csv = csv.DictReader(open(testFile, "r"))
	test_file_csv_data = list(test_file_csv)
	ims_dict = {}
	for i in range(12):
		ims_dict[i] = []
	for i in range(len(test_file_csv_data)):
		row = test_file_csv_data[i]
		cl_id = int(row['Class ID'])
		obj = int(row['Object'])
		if obj == obj_dict[cl_id]:
			ims_dict[cl_id].append(i)
	return ims_dict


if __name__ == "__main__":
	dirName = "byol_object_final_1"
	start_time = time.time()
	backbone = models.resnet18(pretrained = False, num_classes = 10)
	backbone.fc = nn.Identity()
	src_file_name = "../output/" + dirName + "/unsupervised_backbone_100.pt"
	test_pred_file = "../output/" + dirName + "/test_predictions_0.csv"
	test_data_file = "../data/toybox_data_interpolated_cropped_test.csv"
	sel_objs = get_first_object(test_data_file)
	sel_ims = get_selected_ims(testFile = test_data_file, obj_dict = sel_objs)
	bool_list = get_correct_list(srcFile = test_pred_file)
	inv_bool_list = np.asarray([not elem for elem in bool_list])
	bool_list = np.asarray(bool_list)
	weights_dict = torch.load(src_file_name)
	new_state_dict = OrderedDict()
	for k, v in weights_dict.items():
		name = k[8:]
		new_state_dict[name] = v
	backbone.load_state_dict(new_state_dict)

	train_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.ToTensor(),
										 transforms.Normalize(mean, std)])
	rng = np.random.default_rng(0)
	trainData = dataloader_umap.data_loader(root = "../data/", rng = rng, train = False,
												  hypertune = False,
												  transform = train_transform, fraction = 0.5)

	trainLoader = torch.utils.data.DataLoader(trainData, batch_size = 512, shuffle = False,
											  num_workers = 4, pin_memory = False,
											  persistent_workers = True)
	print("Number of Training Examples:", len(trainData))
	backbone = backbone.cuda()
	all_logits = None
	all_labels = None
	for _, images, labels in trainLoader:
		images = images.cuda()
		with torch.no_grad():
			logits = backbone.forward(images)
			if all_logits is None:
				all_logits = logits.detach().cpu().numpy()
				all_labels = np.asarray(labels)
			else:
				logits = logits.detach().cpu().numpy()
				all_logits = np.concatenate((all_logits, logits), axis = 0)
				all_labels = np.concatenate((all_labels, np.asarray(labels)), axis = 0)
	print(all_logits.shape, all_labels.shape, time.time() - start_time)
	nns = [10, 25, 50, 100, 200, 400]
	mds = [0.02, 0.04, 0.08, 0.1, 0.2, 0.4]
	# nns = [10, 25]
	# mds = [0.02, 0.04]
	for nn in nns:
		for md in mds:
			print("Starting with n_neighbors = {:d} and min_dist = {:0.2f}".format(nn, md))
			reducer = umap.UMAP(random_state = 42, n_neighbors = nn, min_dist = md, metric = 'cosine')
			embedding = reducer.fit_transform(all_logits)
			print(embedding.shape)
			print("Total Time:", time.time() - start_time)
			cols = np.linspace(0.0, 1.0, 12)
			cmap = cm.get_cmap('nipy_spectral')
			for i in range(12):
				obj_embedding = embedding[sel_ims[i], :]
				obj_bool_list = bool_list[sel_ims[i]]
				obj_inv_bool_list = inv_bool_list[sel_ims[i]]
				correct = sum(obj_bool_list)
				incorrect = len(obj_embedding) - correct
				# print(i, correct, incorrect, len(obj_embedding))
				plt.scatter(obj_embedding[obj_bool_list, 0], obj_embedding[obj_bool_list, 1], c = [cmap(cols[i]) for _ in range(correct)],
							cmap = 'nipy_spectral', label = classes[i], marker = '.', s = 200)

				plt.annotate(classes[i], (obj_embedding[0][0], obj_embedding[0][1]), weight = 'bold', fontsize = 25)

				plt.scatter(obj_embedding[obj_inv_bool_list, 0], obj_embedding[obj_inv_bool_list, 1],
							c = [cmap(cols[i]) for _ in range(incorrect)], cmap = 'nipy_spectral', marker = 'x', s = 200)
			plt.gca().set_aspect('auto', 'datalim')
			plt.legend(loc = 'lower right', prop = {'size' : 25})
			fig = plt.gcf()
			fig.set_size_inches(20, 20)
			outDir = "../output/" + dirName + "/umaps"
			if not os.path.isdir(outDir):
				os.mkdir(outDir)
			outPath = "../output/" + dirName + "/umaps/umap_" + str(nn) + "_" + str(md) + ".png"
			plt.savefig(outPath)
			plt.close()
