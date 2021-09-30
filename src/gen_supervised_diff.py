import csv
import pickle
import argparse
import os
import numpy as np
import time

outDir = "../supervised_data/"
imgs_file_name = "../supervised_data/toybox_interpolated_cropped_all.pickle"
csv_file_name = "../supervised_data/toybox_interpolated_cropped_all.csv"
classes = ['airplane', 'ball', 'car', 'cat' , 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon', 'truck']
DEF_TEST_NO = {
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


def get_parser():
	parser = argparse.ArgumentParser(description = "")
	parser.add_argument("--default", "-d", default = False, action = "store_true")
	parser.add_argument("--seed", "-s", default = -1, type = int)
	parser.add_argument("--saveDir", "-sd", required = True, type = str)
	return parser.parse_args()


def get_test_items(rng):
	test_items = {}
	for cl in classes:
		items = rng.choice(a = np.arange(1, 31), size = 3, replace = False)
		test_items[cl] = items
	return test_items


def gen_train_test(test_items, saveDir):
	start_time = time.time()
	data_csv = list(csv.DictReader(open(csv_file_name, "r")))
	images_pickle = pickle.load(open(imgs_file_name, "rb"))
	train_num = 0
	test_num = 0
	train_imgs = []
	test_imgs = []
	train_csv_name = saveDir + "train.csv"
	train_pickle_name = saveDir + "train.pickle"
	test_csv_name = saveDir + "test.csv"
	test_pickle_name = saveDir + "test.pickle"
	train_csv_file = open(train_csv_name, "w")
	test_csv_file = open(test_csv_name, "w")
	train_pickle_file = open(train_pickle_name, "wb")
	test_pickle_file = open(test_pickle_name, "wb")
	train_csv = csv.writer(train_csv_file)
	test_csv = csv.writer(test_csv_file)
	train_csv.writerow(["Index", "Class", "Class ID", "Object", "Transformation", "File Name"])
	test_csv.writerow(["Index", "Class", "Class ID", "Object", "Transformation", "File Name"])

	for i in range(10):#len(data_csv)):
		row = data_csv[i]
		im = images_pickle[i]
		cl = row['Class']
		obj = int(row['Object'])
		cl_id = row['Class ID']
		tr = row['Transformation']
		fName = row['File Name']
		if obj in test_items[cl]:
			test_imgs.append(im)
			test_csv.writerow([test_num, cl, cl_id, obj, tr, fName])
			test_num += 1
		else:
			train_imgs.append(im)
			train_csv.writerow([train_num, cl, cl_id, obj, tr, fName])
			train_num += 1
		print(train_num, test_num)
	pickle.dump(train_imgs, train_pickle_file, pickle.DEFAULT_PROTOCOL)
	pickle.dump(test_imgs, test_pickle_file, pickle.DEFAULT_PROTOCOL)
	train_csv_file.close()
	train_pickle_file.close()
	test_csv_file.close()
	test_pickle_file.close()
	print(train_num, test_num, len(data_csv), "Total time:", time.time() - start_time)


if __name__ == "__main__":
	data_args = vars(get_parser())
	assert os.path.isdir(outDir)
	data_args['saveDir'] = outDir + data_args['saveDir'] + "/"
	try:
		assert not os.path.isdir(data_args['saveDir'])
	except AssertionError:
		raise RuntimeError("Provided save directory already exists. Provide new name.")
	os.mkdir(data_args['saveDir'])
	if data_args['default']:
		test_no = DEF_TEST_NO
	else:
		try:
			assert data_args['seed'] != -1
		except AssertionError:
			raise AssertionError("Provide seed for test items using -s flag")
		random_gen = np.random.default_rng(data_args['seed'])
		test_no = get_test_items(rng = random_gen)
	print(data_args['saveDir'], test_no)
	gen_train_test(test_items = test_no, saveDir = data_args['saveDir'])
