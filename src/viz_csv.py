import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import numpy as np

plt.rcParams['figure.figsize'] = (12, 20)

classes = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon', 'truck']
np.set_printoptions(precision = 3)
np.set_printoptions(suppress = True)
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

vids = ["rxminus", "rxplus", "ryminus", "ryplus", "rzminus", "rzplus"]


def build_conf_matrix_obj(file_name, test_data_file_name, out_dir):
	conf_mat = np.zeros((3 * len(classes), len(classes)), dtype = np.float32)
	total_count = np.zeros(3 * len(classes))
	test_items = {}
	for cl in classes:
		test_items[cl] = []
	with open(file_name, "r") as test_pred_file:
		eval_data = list(csv.DictReader(test_pred_file))
	with open(test_data_file_name, "r") as test_data_file:
		test_data = list(csv.DictReader(test_data_file))
	for i in range(len(eval_data)):
		idx = int(eval_data[i]['Index'])
		true_label = int(eval_data[i]['True Label'])
		pred_label = int(eval_data[i]['Predicted Label'])
		obj = int(test_data[idx]['Object'])
		cl = test_data[idx]['Class']
		if obj not in test_items[cl]:
			test_items[cl].append(obj)
		row = true_label * 3 + test_items[cl].index(obj)
		total_count[row] += 1
		conf_mat[row][pred_label] += 1
	for i in range(3 * len(classes)):
		for j in range(len(classes)):
			conf_mat[i][j] /= total_count[i]
			conf_mat[i][j] *= 100
	im = plt.imshow(conf_mat, vmin = 0, vmax = 100.0, cmap = cm.get_cmap('cividis'), aspect = 'auto')
	plt.colorbar(im)
	plt.xlabel("Predicted Labels", fontsize = 20)
	plt.ylabel("True Labels", fontsize = 20)
	for i in range(3*len(classes)):
		for j in range(len(classes)):
			plt.text(j, i, str(round(conf_mat[i][j], 2)), fontsize = 14, ha = 'center', va = 'center',
					 weight = 'bold')
	plt.yticks([3*i + 1 for i in range(12)], classes, fontsize = 15)

	plt.xticks([i for i in range(12)], classes, va = "top", fontsize = 15)

	fig = plt.gcf()
	fig.set_size_inches(30, 20)
	# plt.show()
	plt.savefig(out_dir + "conf_matrix_objects.png", dpi = 100)
	plt.close()


def build_conf_matrix(file_name, outdir):
	conf_mat = np.zeros((len(classes), len(classes)), dtype = np.float32)
	total_count = np.zeros(len(classes))
	with open(file_name, "r") as test_pred_file:
		eval_data = list(csv.DictReader(test_pred_file))
	for i in range(len(eval_data)):
		true_label = int(eval_data[i]['True Label'])
		pred_label = int(eval_data[i]['Predicted Label'])
		total_count[true_label] += 1
		conf_mat[true_label][pred_label] += 1
	for i in range(len(classes)):
		for j in range(len(classes)):
			conf_mat[i][j] /= total_count[i]
			conf_mat[i][j] *= 100
	im = plt.imshow(conf_mat, vmin = 0, vmax = 100.0, cmap = cm.get_cmap('cividis'))
	plt.colorbar(im)
	plt.xlabel("Predicted Labels", fontsize = 20)
	plt.ylabel("True Labels", fontsize = 20)
	for i in range(len(classes)):
		for j in range(len(classes)):
			plt.text(j, i, str(round(conf_mat[i][j], 2)), fontsize = 17, ha = 'center', va = 'center', weight = 'bold')
	plt.yticks([i for i in range(12)], classes, fontsize = 15)
	plt.xticks([i for i in range(12)], classes, va = "top", fontsize = 15)
	fig = plt.gcf()
	fig.set_size_inches(30, 20)
	plt.savefig(outdir + "conf_matrix.png", dpi = 100)
	plt.close()


def calc_total_accuracy(file_name):
	test_correct = 0
	test_total = 0
	with open(file_name, "r") as test_pred_file:
		eval_data = list(csv.DictReader(test_pred_file))
	for i in range(len(eval_data)):
		pred_label = int(eval_data[i]['Predicted Label'])
		true_label = int(eval_data[i]['True Label'])
		if pred_label == true_label:
			test_correct += 1
		test_total += 1
	print("Overall accuracy is {:2.2f}".format(test_correct / test_total * 100))


def split_eval_by_classes(file_name):
	test_correct = [0 for _ in range(len(classes))]
	test_total = [0 for _ in range(len(classes))]
	with open(file_name, "r") as test_pred_file:
		eval_data = list(csv.DictReader(test_pred_file))
	for i in range(len(eval_data)):
		pred_label = int(eval_data[i]['Predicted Label'])
		true_label = int(eval_data[i]['True Label'])
		if pred_label == true_label:
			test_correct[true_label] += 1
		test_total[true_label] += 1
	print("==================================================================================")
	print("Printing accuracy for each class")
	print("==================================================================================")
	for i in range(len(classes)):
		print("{0:12s}:  {1:2.2f}".format(classes[i], test_correct[i] / test_total[i] * 100))
	print("==================================================================================")


def split_eval_by_objects(file_name, testDataFile):
	gantt_items = []
	test_correct = {}
	test_total = {}
	test_items = {}
	for cl in classes:
		test_items[cl] = []
	with open(file_name, "r") as test_pred_file:
		eval_data = list(csv.DictReader(test_pred_file))
	with open(testDataFile, "r") as test_data_file:
		test_data = list(csv.DictReader(test_data_file))
	for i in range(len(eval_data)):
		idx = int(eval_data[i]['Index'])
		cl = test_data[idx]['Class']
		pred_label = int(eval_data[i]['Predicted Label'])
		true_label = int(eval_data[i]['True Label'])
		obj = int(test_data[idx]['Object'])
		if obj not in test_items[cl]:
			test_items[cl].append(obj)
		assert int(true_label) == int(test_data[int(idx)]['Class ID'])
		if (true_label, obj) not in test_correct.keys():
			test_correct[(true_label, obj)] = 0
			test_total[(true_label, obj)] = 0
		if pred_label == true_label:
			test_correct[(true_label, obj)] += 1
		test_total[(true_label, obj)] += 1
	print(test_items)
	print("==================================================================================")
	print("Printing accuracy for each class and object")
	print("==================================================================================")
	print("{0:12s}:  {1:5s}    {2:5s}    {3:5s}    {4:5s}".format("Class", "Obj1", "Obj2", "Obj3", "Total"))
	for i in range(len(classes)):
		cl = classes[i]
		corr = [0.0 for i in range(len(VAL_NO['ball']))]
		tot = 0
		tot_corr = 0
		for j in range(len(test_items['ball'])):
			test_obj = test_items[cl][j]
			tot += test_total[(i, test_obj)]
			tot_corr += test_correct[(i, test_obj)]
			corr[j] = test_correct[(i, test_obj)] / test_total[(i, test_obj)] * 100
			if corr[j] < 10:
				gantt_items.append((cl, test_obj))
		pc_corr = tot_corr / tot * 100
		print("{0:12s}:  {1:2.2f}    {2:2.2f}    {3:2.2f}    {4:2.2f}".format(classes[i],
				corr[0], corr[1], corr[2], pc_corr))
	print("==================================================================================")
	return gantt_items


def generate_gantt(test_data_file, cl, obj, file_name, out_dir):
	with open(test_data_file, "r") as ff:
		test_data_csv = list(csv.DictReader(ff))

	with open(file_name, "r") as ff:
		rr = list(csv.DictReader(ff))

	tot_test = [0 for _ in range(len(classes))]
	tot_correct = [0 for _ in range(len(classes))]
	cnt = [0 for _ in vids]
	cnt_arr = [[] for _ in vids]
	for row in range(len(rr)):
		idx = rr[row]['Index']
		true_label = rr[row]['True Label']
		pred_label = rr[row]['Predicted Label']
		assert int(true_label) == int(test_data_csv[int(idx)]['Class ID'])
		test_obj = int(test_data_csv[int(idx)]['Object'])
		test_tr = test_data_csv[int(idx)]['Transformation']
		if int(true_label) == classes.index(cl) and test_obj == obj:
			if test_tr in vids:
				vid_idx = vids.index(test_tr)
				if int(true_label) == int(pred_label):
					cnt_arr[vid_idx].append(True)
				else:
					cnt_arr[vid_idx].append(False)
				cnt[vid_idx] += 1
		tot_test[int(true_label)] += 1
		if true_label == pred_label:
			tot_correct[int(true_label)] += 1

	fig, gnt = plt.subplots()
	gnt.set_ylim(-5, 90)
	width = 10
	for j in range(len(vids)):
		start = 0
		for i in range(len(cnt_arr[j])):
			if cnt_arr[j][i]:
				gnt.broken_barh([(start, width), ], (j*15, 10), facecolors = 'tab:green')
			else:
				gnt.broken_barh([(start, width), ], (j*15, 10), facecolors = 'tab:red')
			gnt.broken_barh([(start + width, 3), ], (j*15, 10), facecolors = 'black')
			start += width + 3
	plt.yticks([15 * i + 5 for i in range(len(vids))], vids, fontsize = 20)
	fig = plt.gcf()
	fig.set_size_inches(30, 20)
	fr = plt.gca()
	fr.get_xaxis().set_visible(False)
	plt.savefig(out_dir + cl + "_" + str(obj) + ".png")
	plt.close()
	# plt.show()


def multiple_gantt(test_data_file, cl, obj, file_names, out_dir):
	with open(test_data_file, "r") as ff:
		test_data_csv = list(csv.DictReader(ff))
	file_name = file_names[0]
	with open(file_name, "r") as ff:
		rr = list(csv.DictReader(ff))

	tot_test = [0 for _ in range(len(classes))]
	tot_correct = [0 for _ in range(len(classes))]
	cnt = [0 for _ in vids]
	cnt_arr = [[] for _ in vids]
	for row in range(len(rr)):
		idx = rr[row]['Index']
		true_label = rr[row]['True Label']
		pred_label = rr[row]['Predicted Label']
		assert int(true_label) == int(test_data_csv[int(idx)]['Class ID'])
		test_obj = int(test_data_csv[int(idx)]['Object'])
		test_tr = test_data_csv[int(idx)]['Transformation']
		if int(true_label) == classes.index(cl) and test_obj == obj:
			if test_tr in vids:
				vid_idx = vids.index(test_tr)
				if int(true_label) == int(pred_label):
					cnt_arr[vid_idx].append(True)
				else:
					cnt_arr[vid_idx].append(False)
				cnt[vid_idx] += 1
		tot_test[int(true_label)] += 1
		if true_label == pred_label:
			tot_correct[int(true_label)] += 1

	file_name = file_names[1]
	with open(file_name, "r") as ff:
		rr = list(csv.DictReader(ff))

	tot_test_2 = [0 for _ in range(len(classes))]
	tot_correct_2 = [0 for _ in range(len(classes))]
	cnt_2 = [0 for _ in vids]
	cnt_arr_2 = [[] for _ in vids]
	for row in range(len(rr)):
		idx = rr[row]['Index']
		true_label = rr[row]['True Label']
		pred_label = rr[row]['Predicted Label']
		assert int(true_label) == int(test_data_csv[int(idx)]['Class ID'])
		test_obj = int(test_data_csv[int(idx)]['Object'])
		test_tr = test_data_csv[int(idx)]['Transformation']
		if int(true_label) == classes.index(cl) and test_obj == obj:
			if test_tr in vids:
				vid_idx = vids.index(test_tr)
				if int(true_label) == int(pred_label):
					cnt_arr_2[vid_idx].append(True)
				else:
					cnt_arr_2[vid_idx].append(False)
				cnt_2[vid_idx] += 1
		tot_test_2[int(true_label)] += 1
		if true_label == pred_label:
			tot_correct_2[int(true_label)] += 1

	fig, gnt = plt.subplots()
	gnt.set_ylim(-5, 120)
	width = 10
	for j in range(len(vids)):
		start = 0
		for i in range(len(cnt_arr[j])):
			if cnt_arr[j][i]:
				gnt.broken_barh([(start, width), ], (j*20, 7), facecolors = 'tab:green')
			else:
				gnt.broken_barh([(start, width), ], (j*20, 7), facecolors = 'tab:red')

			gnt.broken_barh([(start + width, 3), ], (j*20, 7), facecolors = 'black')
			if cnt_arr_2[j][i]:
				gnt.broken_barh([(start, width), ], (j*20 + 8, 7), facecolors = 'tab:green')
			else:
				gnt.broken_barh([(start, width), ], (j*20 + 8, 7), facecolors = 'tab:red')

			gnt.broken_barh([(start + width, 3), ], (j*20 + 8, 7), facecolors = 'black')
			start += width + 3
	plt.yticks([20 * i + 7.5 for i in range(len(vids))], vids, fontsize = 20)
	fig = plt.gcf()
	fig.set_size_inches(30, 20)
	fr = plt.gca()
	fr.get_xaxis().set_visible(False)
	plt.savefig(out_dir + cl + "_" + str(obj) + "_multiple.png")
	plt.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "")
	parser.add_argument("--dir", "-d", required = True, type = str)
	parser.add_argument("--val", "-v", default = False, action = 'store_true')
	parser = vars(parser.parse_args())
	directory = "../output/" + parser['dir'] + "/"
	fileName = directory + "test_predictions_0.csv"
	fileName2 = "../output/" + parser['dir'] + "_2/test_predictions_0.csv"
	if parser['val']:
		testFileName = "../data/toybox_data_cropped_val.csv"
	else:
		testFileName = "../data/toybox_data_cropped_test.csv"
	build_conf_matrix(file_name = fileName, outdir = directory)
	build_conf_matrix_obj(file_name = fileName, test_data_file_name = testFileName, out_dir = directory)
	calc_total_accuracy(file_name = fileName)
	split_eval_by_classes(file_name = fileName)
	low_acc_items = split_eval_by_objects(file_name = fileName, testDataFile = testFileName)
	for cat, obj_num in low_acc_items:
		generate_gantt(test_data_file = testFileName, file_name = fileName, cl = cat, obj = obj_num, out_dir = directory)
		# multiple_gantt(test_data_file = testFileName, file_names = [fileName, fileName2], cl = cat, obj = obj_num,
		# out_dir = directory)
