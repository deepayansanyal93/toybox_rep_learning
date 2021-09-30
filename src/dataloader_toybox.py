import torch
from data_loader import data_loader_generic


classes = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon', 'truck']

TEST_NO = {
	'ball'      : [1, 7, 9],
	'spoon'     : [5, 7, 8],
	'mug'       : [12, 13, 14],
	'cup'       : [12, 13, 15],
	'giraffe'   : [1, 5, 13],
	'horse'     : [1, 10, 15],
	'cat'       : [4, 9, 15],
	'duck'      : [5, 9, 13],
	'helicopter': [5, 10, 15],
	'airplane'  : [2, 6, 15],
	'truck'     : [2, 6, 8],
	'car'       : [6, 11, 13],
}

VAL_NO = {
	'airplane': [30, 29, 28],
	'ball': [30, 29, 28],
	'car': [30, 29, 28],
	'cat': [30, 29, 28],
	'cup': [30, 29, 28],
	'duck': [30, 29, 28],
	'giraffe': [30, 29, 28],
	'helicopter': [30, 29, 28],
	'horse': [30, 29, 28],
	'mug': [30, 29, 28],
	'spoon': [30, 29, 28],
	'truck': [30, 29, 28]
}


class dataloader_toybox(data_loader_generic):

	def __init__(self, root, rng, train = True, transform = None, nViews = 2, size = 224, split =
				"unsupervised", fraction = 1.0, distort = 'self', adj = -1, hyperTune = True, distortArg = False,
				 interpolate = False):
		self.tr_start_key = 'Tr Start'
		self.tr_end_key = 'Tr End'
		self.obj_start_key = 'Obj Start'
		self.obj_end_key = 'Obj End'
		self.tr_key = 'Transformation'
		self.cl_start_key = 'CL Start'
		self.cl_end_key = 'CL End'
		if not hyperTune:
			if not interpolate:
				self.trainImagesFile = "../data/toybox_data_cropped_train.pickle"
				self.trainLabelsFile = "../data/toybox_data_cropped_train.csv"
				self.testImagesFile = "../data/toybox_data_cropped_test.pickle"
				self.testLabelsFile = "../data/toybox_data_cropped_test.csv"
			else:
				self.trainImagesFile = "../data2/toybox_data_interpolated_cropped_train.pickle"
				self.trainLabelsFile = "../data2/toybox_data_interpolated_cropped_train.csv"
				self.testImagesFile = "../data2/toybox_data_interpolated_cropped_test.pickle"
				self.testLabelsFile = "../data2/toybox_data_interpolated_cropped_test.csv"
		else:
			if not interpolate:
				self.trainImagesFile = "../data/toybox_data_cropped_dev.pickle"
				self.trainLabelsFile = "../data/toybox_data_cropped_dev.csv"
				self.testImagesFile = "../data/toybox_data_cropped_val.pickle"
				self.testLabelsFile = "../data/toybox_data_cropped_val.csv"
			else:
				self.trainImagesFile = "../data2/toybox_data_interpolated_cropped_dev.pickle"
				self.trainLabelsFile = "../data2/toybox_data_interpolated_cropped_dev.csv"
				self.testImagesFile = "../data2/toybox_data_interpolated_cropped_val.pickle"
				self.testLabelsFile = "../data2/toybox_data_interpolated_cropped_val.csv"

		super().__init__(root = root, rng = rng, train = train, transform = transform, nViews = nViews, size = size,
						 split = split, fraction = fraction, distort = distort, adj = adj, hyperTune = hyperTune,
						 distortArg = distortArg)
