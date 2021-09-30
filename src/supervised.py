import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import tqdm
import argparse
import os
import csv

import dataloader_supervised
import utils

mean = (0.3499, 0.4374, 0.5199)
std = (0.1623, 0.1894, 0.1775)

outputDir = "../supervised_output/"


def get_parser(desc):
	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument("--batch_size", "-b", default = 64, type = int)
	parser.add_argument("--workers", "-w", default = 1, type = int)
	parser.add_argument("--epochs", "-e", default = 50, type = int)
	parser.add_argument("--lr", "-lr", default = 0.1, type = float)
	parser.add_argument("--hypertune", "-ht", default = False, action = "store_true")
	parser.add_argument("--pretrained", "-pt", default = False, action = 'store_true')
	parser.add_argument("--fraction", "-f", default = 1.0, type = float)
	parser.add_argument("--save", "-sv", default = False, action = 'store_true')
	parser.add_argument("--saveName", "-sn", default = "", type = str)
	parser.add_argument("--loadDir", "-ld", default = "", type = str)
	parser.add_argument("--loadFile", "-lf", default = "", type = str)
	parser.add_argument("--finetune", "-ft", default = False, action = 'store_true')
	parser.add_argument("--instance", "-i", default = False, action = 'store_true')
	parser.add_argument("--use-all-instance", "-u", default = True, action = 'store_false')
	return parser.parse_args()


def get_network(pre, num_classes):
	model = torchvision.models.resnet18(pretrained = pre)
	fc_size = model.fc.in_features
	model.fc = nn.Identity()
	classifier = nn.Linear(fc_size, num_classes)
	return model, classifier


def eval_net(network, classifier, train_loader, test_loader, train_file_name = "", test_file_name = ""):
	network.eval()
	top1acc = 0
	top5acc = 0
	totTrainPoints = 0
	if train_file_name == "" or test_file_name == "":
		save = False
	else:
		save = True
	if save:
		train_csv_file = open(train_file_name, "w")
		train_pred_csv = csv.writer(train_csv_file)
		test_csv_file = open(test_file_name, "w")
		test_pred_csv = csv.writer(test_csv_file)
		train_pred_csv.writerow(["Index", "True Label", "Predicted Label"])
		test_pred_csv.writerow(["Index", "True Label", "Predicted Label"])

	for _, (indices, images, labels) in enumerate(train_loader):
		images = images.cuda(non_blocking = True)
		labels = labels.cuda(non_blocking = True)
		with torch.no_grad():
			feats = network.forward(images)
			logits = classifier(feats)
		top, pred = utils.calc_accuracy(logits, labels, topk = (1, 5))
		top1acc += top[0].item() * pred.shape[0]
		top5acc += top[1].item() * pred.shape[0]
		totTrainPoints += pred.shape[0]
		if save:
			pred, labels, indices = pred.cpu().numpy(), labels.cpu().numpy(), indices.cpu().numpy()
			for idx in range(pred.shape[0]):
				row = [indices[idx], labels[idx], pred[idx]]
				train_pred_csv.writerow(row)
	top1acc /= totTrainPoints
	top5acc /= totTrainPoints
	print("Train Accuracies 1 and 5:", top1acc, top5acc)

	top1corr = 0
	top5acc = 0
	totTestPoints = 0
	for _, (indices, images, labels) in enumerate(test_loader):
		images = images.cuda(non_blocking = True)
		labels = labels.cuda(non_blocking = True)
		with torch.no_grad():
			feats = network.forward(images)
			logits = classifier.forward(feats)
		top, pred = utils.calc_accuracy(logits, labels, topk = (1, 5))
		top1corr += top[0].item() * indices.size()[0]
		top5acc += top[1].item() * indices.size()[0]
		totTestPoints += indices.size()[0]
		if save:
			pred, labels, indices = pred.cpu().numpy(), labels.cpu().numpy(), indices.cpu().numpy()
			for idx in range(pred.shape[0]):
				row = [indices[idx], labels[idx], pred[idx]]
				test_pred_csv.writerow(row)

	top1acc = top1corr / totTestPoints
	top5acc /= totTestPoints
	print("Test Accuracies 1 and 5:", top1acc, top5acc)
	if save:
		train_csv_file.close()
		test_csv_file.close()
	network.train()


def train(network, classifier, train_data, test_data, args):
	gpu = 0
	torch.cuda.set_device(gpu)
	network.cuda(gpu)
	classifier.cuda(gpu)

	trainLoader = torch.utils.data.DataLoader(train_data, batch_size = args['batch_size'], shuffle = False,
											  num_workers = args['workers'], pin_memory = False,
											  persistent_workers = True)

	testLoader = torch.utils.data.DataLoader(test_data, batch_size = args['batch_size'], shuffle = False,
											 pin_memory = False, num_workers = args['workers'])

	optimizer = optim.Adam(classifier.parameters(), lr = args['lr'], weight_decay = 1e-5)
	if args['finetune']:
		optimizer.add_param_group({'params': network.parameters()})
	else:
		for params in network.parameters():
			params.requires_grad = False
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = max(10, (args['epochs'] - 10)) *
																					len(trainLoader), eta_min = 0.001)
	numEpochs = args['epochs']
	sup_losses = []
	warmup_steps = 10 * len(trainLoader)
	# optimizer.param_groups[0]['lr'] = 1 / warmup_steps * args['lr']
	for ep in range(numEpochs):
		ep_id = 0
		tot_loss = 0
		tqdmBar = tqdm.tqdm(total = len(trainLoader))
		ep_losses = []
		for _, images, labels in trainLoader:
			optimizer.zero_grad()
			images, labels = images.cuda(), labels.cuda()
			feats = network(images)
			logits = classifier(feats)
			loss = nn.CrossEntropyLoss()(logits, labels)
			loss.backward()
			optimizer.step()

			ep_losses.append(loss.item())
			tot_loss += loss.item()
			ep_id += 1
			avg_loss = tot_loss / ep_id
			tqdmBar.update(1)
			tqdmBar.set_description("Epoch: {:d}/{:d} Loss: {:.4f} LR: {:.6f}".format(ep + 1, numEpochs, avg_loss,
																						  optimizer.param_groups[0][
																							  'lr']))
			if ep > 9:
				scheduler.step()
			else:
				curr_step = ep * len(trainLoader) + ep_id + 1
				# optimizer.param_groups[0]['lr'] = curr_step / warmup_steps * args['lr']
		tqdmBar.close()
		sup_losses.append(ep_losses)
		if (ep + 1) % 20 == 0 and ep != numEpochs - 1:

			if args['save']:
				train_csv_file_name = args['saveDir'] + "/train_pred_epoch_" + str(ep + 1) + ".csv"
				test_csv_file_name = args['saveDir'] + "/test_pred_epoch_" + str(ep + 1) + ".csv"
				eval_net(network = network, classifier = classifier, train_loader = trainLoader, test_loader = testLoader,
						 train_file_name = train_csv_file_name, test_file_name = test_csv_file_name)
				fileName = args['saveDir'] + "epoch_" + str(ep + 1) + ".pt"
				torch.save(network.state_dict(), fileName, _use_new_zipfile_serialization = False)
			else:
				eval_net(network = network, classifier = classifier, train_loader = trainLoader, test_loader = testLoader)

	if args['save']:
		train_csv_file_name = args['saveDir'] + "/train_pred_final.csv"
		test_csv_file_name = args['saveDir'] + "/test_pred_final.csv"
		eval_net(network = network, classifier = classifier, train_loader = trainLoader, test_loader = testLoader,
				 train_file_name = train_csv_file_name, test_file_name = test_csv_file_name)
		fileName = args['saveDir'] + "final_model.pt"
		torch.save(network.state_dict(), fileName, _use_new_zipfile_serialization = False)
	else:
		eval_net(network = network, classifier = classifier, train_loader = trainLoader, test_loader = testLoader)


if __name__ == "__main__":
	exp_args = vars(get_parser(desc = "Sup Parser"))
	if not os.path.isdir(outputDir):
		os.mkdir(outputDir)
	if exp_args['save']:
		try:
			assert exp_args['saveName'] != ""
		except AssertionError:
			raise AssertionError("Enter name of directory to save output")
		try:
			assert not os.path.isdir(outputDir + exp_args['saveName'] + "/")
		except AssertionError:
			raise AssertionError("Entered save directory already exists. Enter a new name.")
		exp_args['saveDir'] = outputDir + exp_args['saveName'] + '/'
		os.mkdir(exp_args['saveDir'])
	if exp_args["instance"]:
		net, linear_fc = get_network(pre = False, num_classes = 288)
	else:
		net, linear_fc = get_network(pre = False, num_classes = 12)

	if exp_args['loadDir'] != "":
		if exp_args['loadFile'] == "":
			exp_args['loadFile'] = "final_model.pt"
		targetFileName = outputDir + exp_args['loadDir'] + "/" + exp_args['loadFile']
		try:
			assert os.path.isfile(targetFileName)
		except AssertionError:
			raise AssertionError("Load file could not be found: " + targetFileName)
		net.load_state_dict(torch.load(targetFileName))
	rng = np.random.default_rng(0)
	s = 1
	color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
	train_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224), color_jitter,
										  transforms.RandomCrop(size = 224, padding = 25),
										  transforms.RandomGrayscale(p = 0.2),
										  transforms.RandomHorizontalFlip(p = 0.5),
										  transforms.ToTensor(),
										  transforms.Normalize(mean, std)])
	test_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.ToTensor(),
										 transforms.Normalize(mean, std)])
	if exp_args['instance']:
		if exp_args['use_all_instance']:
			trainData = dataloader_supervised.data_loader(root = "../instance_learning/", rng = rng, train = True,
														  split = "pretraining", transform = train_transform,
														  fraction = exp_args['fraction'], instance = True)

			testData = dataloader_supervised.data_loader(root = "../instance_learning/", rng = rng, train = False,
														 split = "instance", transform = test_transform, instance = True)
		else:
			trainData = dataloader_supervised.data_loader(root = "../instance_learning/", rng = rng, train = True,
														  split = "instance", transform = train_transform,
														  fraction = exp_args['fraction'], instance = True)

			testData = dataloader_supervised.data_loader(root = "../instance_learning/", rng = rng, train = False,
														 split = "instance", transform = test_transform, instance = True)
	else:
		trainData = dataloader_supervised.data_loader(root = "../data/", rng = rng, train = True,
													  hypertune = exp_args['hypertune'],
													  transform = train_transform, fraction = exp_args['fraction'])

		testData = dataloader_supervised.data_loader(root = "../data/", rng = rng, train = False,
													 hypertune = exp_args['hypertune'],
													 transform = test_transform)

	train(network = net, classifier = linear_fc, train_data = trainData, test_data = testData, args = exp_args)
