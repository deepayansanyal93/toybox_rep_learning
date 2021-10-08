import argparse
import torch
import utils
import os
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optimizers
import tqdm

import network_components as ncomponents
from dataloader_toybox import dataloader_toybox as data_toybox

mean = (0.3499, 0.4374, 0.5199)
std = (0.1623, 0.1894, 0.1775)


def get_parser(desc):
	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument("--dataset", "-data", default = "toybox", type = str)
	parser.add_argument("--backbone", "-back", required = True, type = str)
	parser.add_argument("--classifier", "-c", default = "", type = str)
	parser.add_argument("--epochs", "-e", default = 0, type = int)
	parser.add_argument("--lr", "-lr", default = 0.1, type = float)
	parser.add_argument("--ft", "-ft", default = False, action = 'store_true')
	parser.add_argument("--lrs", "-lrs", nargs = '+', type = float)
	parser.add_argument("--batch-size", "-b", default = 256, type = int)
	parser.add_argument("--reps", "-rep", default = 3, type = int)

	return parser.parse_args()


def linear_acc(backbone, classifier, trainLoader, testLoader):
	print("Preparing network for evaluation. Freezing all weights....")
	for params in backbone.parameters():
		params.requires_grad = False
	for params in classifier.parameters():
		params.requires_grad = False
	backbone.eval()
	classifier.eval()

	top1acc = 0
	top5acc = 0
	totTrainPoints = 0
	for _, (indices, images, labels) in enumerate(trainLoader):
		images = images.cuda(non_blocking = True)
		labels = labels.cuda(non_blocking = True)
		with torch.no_grad():
			y = backbone.forward(images)
			logits = classifier.forward(y)
		top, pred = utils.calc_accuracy(logits, labels, topk = (1, 5))
		top1acc += top[0].item() * pred.shape[0]
		top5acc += top[1].item() * pred.shape[0]
		totTrainPoints += pred.shape[0]
	top1acc /= totTrainPoints
	top5acc /= totTrainPoints
	print("Train Accuracies 1 and 5:", top1acc, top5acc)

	top1corr = 0
	top5acc = 0
	totTestPoints = 0
	for _, (indices, images, labels) in enumerate(testLoader):
		images = images.cuda(non_blocking = True)
		labels = labels.cuda(non_blocking = True)
		with torch.no_grad():
			y = backbone.forward(images)
			logits = classifier.forward(y)
		top, pred = utils.calc_accuracy(logits, labels, topk = (1, 5))
		top1corr += top[0].item() * indices.size()[0]
		top5acc += top[1].item() * indices.size()[0]
		totTestPoints += indices.size()[0]
	top1acc = top1corr / totTestPoints
	top5acc /= totTestPoints
	print("Test Accuracies 1 and 5:", top1acc, top5acc)
	return top1acc, top1corr, totTestPoints


def eval_run(args):
	assert os.path.isfile(args["backbone"])
	if args["classifier"] != "":
		assert os.path.isfile(args["classifier"])
	backbone = ncomponents.netBackbone(backbone = "resnet18", model_name = "BYOL backbone")
	backbone.load_state_dict(torch.load(args["backbone"]))
	classifier = nn.Linear(backbone.get_feat_size(), 12)
	if args['classifier'] != "":
		classifier.load_state_dict(torch.load(args["classifier"]))
	backbone = backbone.cuda()
	classifier = classifier.cuda()

	transform_train = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
										  transforms.RandomCrop(padding = 10, size = 224),
										  transforms.ToTensor(), transforms.Normalize(mean, std)])

	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor(),
										 transforms.Normalize(mean, std)])

	rng = np.random.default_rng(0)

	trainSet = data_toybox(root = "./data", train = True, transform = [transform_train, transform_train],
						   split = "super", size = 224,
						   fraction = 0.1, hyperTune = True, rng = rng, interpolate = True)

	testSet = data_toybox(root = "./data", train = False, transform = [transform_test, transform_test], split = "super",
						  size = 224,
						  hyperTune = True, rng = rng, interpolate = True)

	trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = args['batch_size'], shuffle = False, num_workers = 4,
											  pin_memory = False, persistent_workers = True)

	testLoader = torch.utils.data.DataLoader(testSet, batch_size = 128, shuffle = False, pin_memory = False,
											 num_workers = 2)

	for params in backbone.parameters():
		params.requires_grad = False
	backbone.eval()

	num_epochs = args['epochs']
	optimizer = optimizers.SGD(classifier.parameters(), lr = args['lr'], weight_decay = 1e-6, momentum = 0.9)
	for epoch in range(num_epochs):
		tqdmBar = tqdm.tqdm(total = len(trainLoader))
		ep = 0
		tot_loss = 0.0
		for _, images, labels in trainLoader:
			ep += 1
			optimizer.zero_grad()
			images, labels = images.cuda(), labels.cuda()
			feats = backbone.forward(images)
			logits = classifier(feats)
			loss = nn.CrossEntropyLoss()(logits, labels)
			loss.backward()
			optimizer.step()
			tot_loss += loss.item()
			tqdmBar.update(1)
			avg_loss = tot_loss / ep
			tqdmBar.set_description("Epoch: {:d}/{:d} LR: {:f} Loss: {:f}".format(epoch + 1, num_epochs,
																				  optimizer.param_groups[0]['lr'],
																				  avg_loss))

		if epoch % 20 == 19 and ep > 0:
			optimizer.param_groups[0]['lr'] *= 0.9
		tqdmBar.close()

	top1acc, _, _ = linear_acc(backbone = backbone, classifier = classifier, trainLoader = trainLoader, testLoader = testLoader)
	return top1acc


if __name__ == "__main__":
	exp_args = vars(get_parser(desc = "Linear eval"))
	lrs = exp_args['lrs']
	accs = {}
	num_reps = exp_args['reps']
	for i, lr in enumerate(lrs):
		accs[lr] = []
		exp_args['lr'] = lr
		print("================================================================================================")
		print("Starting training with lr {:.3f} ({:d} of {:d} values)".format(lr, i + 1, len(lrs)))
		for j in range(num_reps):
			print("------------------------------------------------------------------------------------------------")
			print("Starting run {:d} of {:d} with lr {:.3f}".format(j + 1, num_reps, lr))
			acc = eval_run(args = exp_args)
			accs[lr].append(acc)
			print("------------------------------------------------------------------------------------------------")
		print("================================================================================================")
	print("lr, mean, std")
	for i, lr in enumerate(lrs):
		print(lr, np.mean(np.asarray(accs[lr])), np.std(np.asarray(accs[lr])))
