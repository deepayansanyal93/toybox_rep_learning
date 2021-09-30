import pickle
import csv
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "")
	parser.add_argument("--dir", "-d", required = True, type = str)
	parser = vars(parser.parse_args())
	dir = "../output/" + parser['dir'] + "/"

	losses_dict = {}
	fileName = dir + "unsupervised_train_losses.pickle"
	with open(fileName, "rb") as ff:
		x = pickle.load(ff)
	print(len(x), len(x[0]))
	for ep in range(len(x)):
		ep_loss = 0
		ep_id = 0
		for i in range(len(x[0])):
			ep_loss += x[ep][i]
			ep_id += 1
		losses_dict[ep + 1] = ep_loss/ep_id
	print(losses_dict)

	fileName = dir + "supervised_losses_0.pickle"
	losses_dict = {}
	with open(fileName, "rb") as ff:
		x = pickle.load(ff)
	print(len(x), len(x[0]))
	for ep in range(len(x)):
		ep_loss = 0
		ep_id = 0
		for i in range(len(x[0])):
			ep_loss += x[ep][i]
			ep_id += 1
		losses_dict[ep + 1] = ep_loss/ep_id
	print(losses_dict)


	fileName = dir + "train_predictions_0.csv"
	with open(fileName, "r") as ff:
		rr = list(csv.DictReader(ff))
	tot_train = 0
	tot_corr = 0
	for row in range(len(rr)):
		tot_train += 1
		if rr[row]['True Label'] == rr[row]['Predicted Label']:
			tot_corr += 1
	print("Train Accuracy:", tot_corr/tot_train * 100)

	fileName = dir + "test_predictions_0.csv"
	with open(fileName, "r") as ff:
		rr = list(csv.DictReader(ff))
	tot_test = 0
	tot_corr = 0
	for row in range(len(rr)):
		tot_test += 1
		if rr[row]['True Label'] == rr[row]['Predicted Label']:
			tot_corr += 1
	print("Test Accuracy:", tot_corr/tot_test * 100)
