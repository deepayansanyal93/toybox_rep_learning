import argparse
import torch
import utils
import os
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

import network_components as ncomponents
from dataloader_toybox import dataloader_toybox as data_toybox

mean = (0.3499, 0.4374, 0.5199)
std = (0.1623, 0.1894, 0.1775)


def get_parser(desc):
	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument("--dataset", "-data", default = "toybox", type = str)
	parser.add_argument("--backbone", "-back", required = True, type = str)
	parser.add_argument("--classifier", "-c", required = True, type = str)
	parser.add_argument("--epochs", "-e", default = 0, type = int)

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


if __name__ == "__main__":
	args = vars(get_parser(desc = "Linear eval"))
	assert os.path.isfile(args["backbone"])
	assert os.path.isfile(args["classifier"])
	backbone = ncomponents.netBackbone(backbone = "resnet18", model_name = "BYOL backbone")
	backbone.load_state_dict(torch.load(args["backbone"]))
	classifier = nn.Linear(backbone.get_feat_size(), 12)
	classifier.load_state_dict(torch.load(args["classifier"]))
	backbone = backbone.cuda()
	classifier = classifier.cuda()

	transform_train = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(padding = 10, size = 224),
										  transforms.ToTensor(), transforms.Normalize(mean, std)])

	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor(),
										 transforms.Normalize(mean, std)])

	rng = np.random.default_rng(0)

	trainSet = data_toybox(root = "./data", train = True, transform = [transform_train, transform_train], split = "super", size = 224,
						   fraction = 0.1, hyperTune = True, rng = rng, interpolate = True)

	testSet = data_toybox(root = "./data", train = False, transform = [transform_test, transform_test], split = "super", size = 224,
						  hyperTune = True, rng = rng, interpolate = True)

	trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = 64, shuffle = False, num_workers = 2,
											  pin_memory = False, persistent_workers = True)

	testLoader = torch.utils.data.DataLoader(testSet, batch_size = 256, shuffle = False, pin_memory = False,
											 num_workers = 2)

	linear_acc(backbone = backbone, classifier = classifier, trainLoader = trainLoader, testLoader = testLoader)
