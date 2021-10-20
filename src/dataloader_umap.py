import cv2
import numpy as np
import torch
import csv
import pickle
import os


class data_loader(torch.utils.data.Dataset):

	def __init__(self, root, rng, train = True, transform = None, size = 224, hypertune = False, fraction = 1.0,
				 instance = False, split = None):

		self.train = train
		self.root = root
		self.transform = transform
		self.hypertune = hypertune
		self.size = size
		self.fraction = fraction
		self.rng = rng
		self.instance = instance
		self.data_path = self.root
		self.split = split
		try:
			assert os.path.isdir(self.data_path)
		except AssertionError:
			raise AssertionError("Data directory not found:", self.data_path)
		if self.instance:
			self.label_key = 'Instance'
			assert self.split == "instance" or self.split == "pretraining"
			if self.split == "instance":
				self.trainImagesFile = self.data_path + "train_instance.pickle"
				self.trainLabelsFile = self.data_path + "train_instance.csv"
				self.testImagesFile = self.data_path + "test_instance.pickle"
				self.testLabelsFile = self.data_path + "test_instance.csv"
			else:
				self.trainImagesFile = self.data_path + "train.pickle"
				self.trainLabelsFile = self.data_path + "train.csv"
				self.testImagesFile = self.data_path + "train.pickle"
				self.testLabelsFile = self.data_path + "train.csv"
		else:
			self.label_key = 'Class ID'
			if self.hypertune:
				self.trainImagesFile = self.data_path + "toybox_data_interpolated_cropped_dev.pickle"
				self.trainLabelsFile = self.data_path + "toybox_data_interpolated_cropped_dev.csv"
				self.testImagesFile = self.data_path + "toybox_data_interpolated_cropped_val.pickle"
				self.testLabelsFile = self.data_path + "toybox_data_interpolated_cropped_val.csv"
			else:
				self.trainImagesFile = self.data_path + "toybox_data_interpolated_cropped_train.pickle"
				self.trainLabelsFile = self.data_path + "toybox_data_interpolated_cropped_train.csv"
				self.testImagesFile = self.data_path + "toybox_data_interpolated_cropped_test.pickle"
				self.testLabelsFile = self.data_path + "toybox_data_interpolated_cropped_test.csv"

		super().__init__()
		if self.train:
			with open(self.trainImagesFile, "rb") as pickleFile:
				self.train_data = pickle.load(pickleFile)
			with open(self.trainLabelsFile, "r") as csvFile:
				self.train_csvFile = list(csv.DictReader(csvFile))
			lenWholeData = len(self.train_data)
			lenTrainData = int(self.fraction * lenWholeData)
			self.indicesSelected = rng.choice(lenWholeData, lenTrainData, replace = False)
		else:
			with open(self.testImagesFile, "rb") as pickleFile:
				self.test_data = pickle.load(pickleFile)
			with open(self.testLabelsFile, "r") as csvFile:
				self.test_csvFile = list(csv.DictReader(csvFile))

	def __len__(self):
		if self.train:
			return len(self.indicesSelected)
		else:
			return len(self.test_data)

	def __getitem__(self, index):
		if self.train:
			actualIndex = self.indicesSelected[index]
			img = np.array(cv2.imdecode(self.train_data[actualIndex], 3))
			label = self.train_csvFile[actualIndex][self.label_key]
		else:
			actualIndex = index
			img = np.array(cv2.imdecode(self.test_data[index], 3))
			label = self.test_csvFile[index][self.label_key]

		if self.transform is not None:
			imgs = self.transform(img)
		else:
			imgs = img
		return actualIndex, imgs, int(label)
