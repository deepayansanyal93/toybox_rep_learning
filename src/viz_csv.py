import csv
import matplotlib.pyplot as plt
import argparse

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

vids = ["rxminus", "rxplus", "ryminus", "ryplus", "rzminus", "rzplus", "hodgepodge"]


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
	test_correct = {}
	test_total = {}
	with open(file_name, "r") as test_pred_file:
		eval_data = list(csv.DictReader(test_pred_file))
	with open(testDataFile, "r") as test_data_file:
		test_data = list(csv.DictReader(test_data_file))
	for i in range(len(eval_data)):
		idx = int(eval_data[i]['Index'])
		pred_label = int(eval_data[i]['Predicted Label'])
		true_label = int(eval_data[i]['True Label'])
		obj = int(test_data[idx]['Object'])
		assert int(true_label) == int(test_data[int(idx)]['Class ID'])
		if (true_label, obj) not in test_correct.keys():
			test_correct[(true_label, obj)] = 0
			test_total[(true_label, obj)] = 0
		if pred_label == true_label:
			test_correct[(true_label, obj)] += 1
		test_total[(true_label, obj)] += 1
	print("==================================================================================")
	print("Printing accuracy for each class and object")
	print("==================================================================================")
	print("{0:12s}:  {1:5s}    {2:5s}    {3:5s}    {4:5s}".format("Class", "Obj1", "Obj2", "Obj3", "Total"))
	for i in range(len(classes)):
		cl = classes[i]
		corr = [0.0 for i in range(len(VAL_NO['ball']))]
		tot = 0
		tot_corr = 0
		for j in range(len(VAL_NO['ball'])):
			test_obj = VAL_NO[cl][j]
			tot += test_total[(i, test_obj)]
			tot_corr += test_correct[(i, test_obj)]
			corr[j] = test_correct[(i, test_obj)] / test_total[(i, test_obj)] * 100
		pc_corr = tot_corr / tot * 100
		print("{0:12s}:  {1:2.2f}    {2:2.2f}    {3:2.2f}    {4:2.2f}".format(classes[i],
				corr[0], corr[1], corr[2], pc_corr))
	print("==================================================================================")


def random_code(test_data_file, cl, obj, tr):
	with open(test_data_file, "r") as ff:
		test_data_csv = list(csv.DictReader(ff))

	with open(fileName, "r") as ff:
		rr = list(csv.DictReader(ff))

	tot_test = [0 for _ in range(len(classes))]
	tot_correct = [0 for _ in range(len(classes))]
	cnt = 0
	cnt_arr = []
	for row in range(len(rr)):
		idx = rr[row]['Index']
		true_label = rr[row]['True Label']
		pred_label = rr[row]['Predicted Label']
		assert int(true_label) == int(test_data_csv[int(idx)]['Class ID'])
		test_obj = int(test_data_csv[int(idx)]['Object'])
		test_tr = test_data_csv[int(idx)]['Transformation']
		if int(true_label) == classes.index(cl) and test_obj == obj and test_tr == tr:
			if int(true_label) == int(pred_label):
				cnt_arr.append(True)
			else:
				cnt_arr.append(False)
			cnt += 1
		tot_test[int(true_label)] += 1
		if true_label == pred_label:
			tot_correct[int(true_label)] += 1
	for i in range(len(classes)):
		print(classes[i], tot_correct[i] / tot_test[i] * 100)
	print(cnt)
	print(cnt_arr)
	fig, gnt = plt.subplots()
	gnt.set_ylim(-30, 30)
	width = 10
	start = 0
	for i in range(len(cnt_arr)):
		if cnt_arr[i]:
			gnt.broken_barh([(start, width), ], (0, 10), facecolors = 'tab:green')
		else:
			gnt.broken_barh([(start, width), ], (0, 10), facecolors = 'tab:red')
		start += width

# plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "")
	parser.add_argument("--dir", "-d", required = True, type = str)
	parser = vars(parser.parse_args())
	directory = "../output/" + parser['dir'] + "/"
	fileName = directory + "test_predictions_0.csv"
	testFileName = "../data/toybox_data_cropped_val.csv"
	calc_total_accuracy(file_name = fileName)
	split_eval_by_classes(file_name = fileName)
	split_eval_by_objects(file_name = fileName, testDataFile = testFileName)
