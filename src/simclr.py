import torch
import torch.optim as optimizers
import torch.nn as nn
import csv
import pickle
import torch.distributed as dist

from network import SimCLRNet
import utils


class SimCLR:
	"""Class implementing specific methods for
	SimCLR learner."""

	def __init__(self, args, network = None, op_classes = 12):
		self.dist = args['num_gpus'] > 0
		if network is None:
			self.network = SimCLRNet(backbone_name = "resnet18", projection_bn = True, projection_relu = True, num_classes
								   = op_classes)
		else:
			self.network = SimCLRNet(backbone_name = "resnet18", projection_bn = True, projection_relu = True, num_classes
								   = op_classes)
		self.args = args
		self.num_epochs_unsupervised = args['epochs1']
		self.linear_ramp_up_epochs = args['rampup']
		try:
			assert self.num_epochs_unsupervised > self.linear_ramp_up_epochs
		except AssertionError:
			raise AssertionError("Number of epochs of unsupervised training has to be greater than 10")
		self.unsupervised_optimizer = optimizers.SGD(self.network.backbone.parameters(), lr = args['lr'], weight_decay =
													 args['weight_decay'], momentum = 0.9)
		self.unsupervised_optimizer.add_param_group({'params': self.network.projection_head.parameters()})

		self.linear_eval_optimizer = optimizers.SGD(self.network.classifier.parameters(), lr = args['lr_ft'],
													weight_decay = args['weight_decay'], momentum = 0.9)

		self.unsupervised_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.unsupervised_optimizer,
																				 T_max = self.num_epochs_unsupervised -
																						 self.linear_ramp_up_epochs,
																				 eta_min = 0.001)
		self.show = False
		self.num_batches = None

	def update_unsupervised_scheduler(self, num_batches):
		self.num_batches = num_batches
		self.unsupervised_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
					self.unsupervised_optimizer, T_max =
					(self.num_epochs_unsupervised - self.linear_ramp_up_epochs) * self.num_batches,
					eta_min = 0.01)

	def save_model(self, file_name):
		self.network.module.save_network(file_name)

	def save_backbone(self, file_name):
		self.network.module.save_backbone(file_name)

	def load_backbone(self, file_name):
		self.network.backbone.load_state_dict(torch.load(file_name))

	def load_classifier(self, file_name):
		self.network.classifier.load_state_dict(torch.load(file_name))

	def save_classifier(self, file_name):
		self.network.module.save_classifier(file_name)

	def get_unsupervised_optimizer(self):
		return self.unsupervised_optimizer

	def get_unsupervised_scheduler(self):
		return self.unsupervised_scheduler

	def get_linear_eval_optimizer(self):
		return self.linear_eval_optimizer

	def get_unsupervised_tqdm_desc(self, ep, numEpochs, loss):
		desc = "Epoch: {:d}/{:d}, Loss: {:.6f}, LR: {:.6f}".format(ep + 1, numEpochs, loss,
																			self.unsupervised_optimizer.param_groups[0][
																							'lr'])

		return desc

	def set_linear_rampup_unsupervised_optimizer(self, ep):
		final_lr = self.args['lr']
		try:
			assert self.num_batches is not None
		except AssertionError:
			raise AssertionError("Update unsupervised_scheduler before starting to train")
		assert ep < self.num_epochs_unsupervised * self.num_batches
		self.unsupervised_optimizer.param_groups[0]['lr'] = \
					(ep + 1) * final_lr / (self.linear_ramp_up_epochs * self.num_batches)


	def prepare_unsupervised(self):
		self.network.unfreeze_backbone()
		self.network.unfreeze_projection()
		self.network.freeze_classifier()
		self.network.print_network_freeze()

	def prepare_linear_eval(self):
		self.network.freeze_backbone()
		self.network.freeze_projection()
		self.network.unfreeze_classifier()
		self.network.print_network_freeze()
		self.network.eval()

	def unsupervised_loss(self, images1, images2, **kwargs):
		torch.autograd.set_detect_anomaly(True)
		images1 = images1.cuda(non_blocking = True)
		images2 = images2.cuda(non_blocking = True)
		gpu = kwargs['gpu']
		all_images = torch.cat((images1, images2), dim = 0)
		features = self.network.module.forward(all_images)
		feats_1 = []
		feats_2 = []
		all_features = [torch.zeros_like(features) for _ in range(self.args['num_gpus'])]

		dist.all_gather(all_features, features)
		all_features[gpu] = features
		for gpu_id in range(self.args['num_gpus']):
			feat_chunks = torch.chunk(all_features[gpu_id], 2)
			feats_1.append(feat_chunks[0])
			feats_2.append(feat_chunks[1])
		feats_1 = torch.cat(feats_1)
		feats_2 = torch.cat(feats_2)
		features = torch.cat((feats_1, feats_2))
		logits, labels = utils.info_nce_loss(features)
		loss = nn.CrossEntropyLoss()(logits, labels)
		return loss

	def linear_eval_loss(self, images, labels):
		images = images.cuda(non_blocking = True)
		labels = labels.cuda(non_blocking = True)
		logits = self.network.module.classify(images)
		loss = nn.CrossEntropyLoss()(logits, labels)
		return loss

	def linear_acc(self, trainLoader, testLoader, run_id = None):
		top1acc = 0
		top5acc = 0
		totTrainPoints = 0
		if self.args["save"]:
			train_pred_file_name = self.args["saveDir"] + "/train_predictions_" + str(run_id) + ".csv"
			train_pred_file = open(train_pred_file_name, "w")
			train_pred_csv = csv.writer(train_pred_file)

			test_pred_file_name = self.args["saveDir"] + "/test_predictions_" + str(run_id) + ".csv"
			test_pred_file = open(test_pred_file_name, "w")
			test_pred_csv = csv.writer(test_pred_file)

			train_indices_file = self.args["saveDir"] + "/train_indices_" + str(run_id) + ".pickle"
			with open(train_indices_file, "wb") as f:
				pickle.dump(trainLoader.dataset.indicesSelected, f, protocol = pickle.DEFAULT_PROTOCOL)

			train_pred_csv.writerow(["Index", "True Label", "Predicted Label"])
			test_pred_csv.writerow(["Index", "True Label", "Predicted Label"])
		for _, (indices, images, labels) in enumerate(trainLoader):
			images = images.cuda(non_blocking = True)
			labels = labels.cuda(non_blocking = True)
			with torch.no_grad():
				logits = self.network.module.classify(images)
			top, pred = utils.calc_accuracy(logits, labels, topk = (1, 5))
			top1acc += top[0].item() * pred.shape[0]
			top5acc += top[1].item() * pred.shape[0]
			totTrainPoints += pred.shape[0]
			if self.args["save"]:
				pred, labels, indices = pred.cpu().numpy(), labels.cpu().numpy(), indices.cpu().numpy()
				for idx in range(pred.shape[0]):
					row = [indices[idx], labels[idx], pred[idx]]
					# print(row)
					train_pred_csv.writerow(row)
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
				logits = self.network.module.classify(images)
			top, pred = utils.calc_accuracy(logits, labels, topk = (1, 5))
			top1corr += top[0].item() * indices.size()[0]
			top5acc += top[1].item() * indices.size()[0]
			totTestPoints += indices.size()[0]
			if self.args["save"]:
				pred, labels, indices = pred.cpu().numpy(), labels.cpu().numpy(), indices.cpu().numpy()
				for idx in range(pred.shape[0]):
					row = [indices[idx], labels[idx], pred[idx]]
					test_pred_csv.writerow(row)
		top1acc = top1corr / totTestPoints
		top5acc /= totTestPoints

		if self.args["save"]:
			train_pred_file.close()
			test_pred_file.close()

		print("Test Accuracies 1 and 5:", top1acc, top5acc)
		return top1acc, top1corr, totTestPoints
