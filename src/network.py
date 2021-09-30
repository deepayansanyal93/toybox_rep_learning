import torch.nn as nn
import torch
import copy
import numpy as np
import torchvision.models as models

import network_components as ncomponents
import network_utils as nutils


def init_weights(m):
	if type(m) == nn.Linear or type(m) == nn.Conv2d:
		nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			m.bias.data.fill_(0.01)


class SimCLRNet(nn.Module):
	"""Network architecture for SimCLR experiments
	"""

	def __init__(self, backbone_name, projection_bn, projection_relu, num_classes = 12):
		super().__init__()
		self.backbone_name = backbone_name
		self.projection_bn = projection_bn
		self.projection_relu = projection_relu
		self.num_classes = num_classes
		self.backbone = ncomponents.netBackbone(backbone = backbone_name, model_name = "SimCLR backbone")
		backbone_feat_size = self.backbone.get_feat_size()
		proj_mlp_shape = [backbone_feat_size, backbone_feat_size * 2, 128]

		self.projection_head = ncomponents.MLP(num_neurons = proj_mlp_shape, relu_in_middle = self.projection_relu,
											   bn_in_middle = self.projection_bn,
											   model_name = "SimCLR projection head")

		self.classifier = nn.Linear(backbone_feat_size, num_classes)
		self.backbone.apply(init_weights)
		self.projection_head.apply(init_weights)
		self.classifier.apply(init_weights)
		self.unsupervised = True

	def forward(self, x):
		y = self.backbone(x)
		y = self.projection_head(y)
		return y

	def classify(self, x):
		y = self.backbone(x)
		y = self.classifier(y)
		return y

	def save_backbone(self, file_name):
		print("Saving SimCLR backbone to", file_name)
		torch.save(self.backbone.state_dict(), file_name, _use_new_zipfile_serialization = False)

	def save_classifier(self, file_name):
		print("Saving classifier to", file_name)
		torch.save(self.classifier.state_dict(), file_name, _use_new_zipfile_serialization = False)

	def save_network(self, fileName):
		print("Saving SimCLR network to", fileName)
		torch.save(self.state_dict(), fileName, _use_new_zipfile_serialization = False)

	def freeze_all_weights(self):
		self.freeze_backbone()
		self.freeze_projection()
		self.freeze_classifier()

	def freeze_backbone(self):
		for params in self.backbone.parameters():
			params.requires_grad = False

	def unfreeze_backbone(self):
		for params in self.backbone.parameters():
			params.requires_grad = True

	def freeze_classifier(self):
		for params in self.classifier.parameters():
			params.requires_grad = False

	def unfreeze_classifier(self):
		for params in self.classifier.parameters():
			params.requires_grad = True

	def freeze_projection(self):
		for params in self.projection_head.parameters():
			params.requires_grad = False

	def unfreeze_projection(self):
		for params in self.projection_head.parameters():
			params.requires_grad = True

	def print_network_freeze(self):
		backbone_total_params = sum(p.numel() for p in self.backbone.parameters())
		backbone_trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
		projection_total_params = sum(p.numel() for p in self.projection_head.parameters())
		projection_trainable_params = sum(p.numel() for p in self.projection_head.parameters() if p.requires_grad)
		classifier_total_params = sum(p.numel() for p in self.classifier.parameters())
		classifier_trainable_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
		print("{}/{} parameters in backbone are trainable.".format(backbone_trainable_params, backbone_total_params))
		print("{}/{} parameters in projection head are trainable.".format(projection_trainable_params, projection_total_params))
		print("{}/{} parameters in classifier are trainable.".format(classifier_trainable_params, classifier_total_params))


class BYOLNet(nn.Module):
	"""Network architecture for BYOL experiments
	"""

	def __init__(self, backbone_name, projection_bn, projection_relu, num_classes = 12, beta = 0.996):
		super().__init__()
		self.backbone_name = backbone_name
		self.projection_relu = projection_relu
		self.projection_bn = projection_bn
		self.num_classes = num_classes
		self.start_beta = beta
		self.beta = beta
		self.backbone = ncomponents.netBackbone(backbone = backbone_name, model_name = "BYOL backbone")
		backbone_feat_size = self.backbone.get_feat_size()
		self.backbone_feat_size = backbone_feat_size

		proj_mlp_shape = [backbone_feat_size, backbone_feat_size * 2, 128]
		pred_mlp_shape = [128, backbone_feat_size * 2, 128]
		self.encoder_projection = ncomponents.MLP(num_neurons = proj_mlp_shape, relu_in_middle = self.projection_relu,
											   bn_in_middle = self.projection_bn,
											   model_name = "BYOL encoder projection")
		self.encoder_prediction = ncomponents.MLP(num_neurons = pred_mlp_shape, relu_in_middle = self.projection_relu,
											   bn_in_middle = self.projection_bn, model_name = "BYOL encoder prediction")
		self.classifier = nn.Linear(backbone_feat_size, num_classes)
		self.backbone.apply(nutils.init_weights)
		self.encoder_projection.apply(nutils.init_weights)
		self.encoder_prediction.apply(nutils.init_weights)
		self.classifier.apply(nutils.init_weights)

		self.target_backbone = copy.deepcopy(self.backbone)
		self.target_projection = copy.deepcopy(self.encoder_projection)

		self.unsupervised = True

	def save_backbone(self, fileName):
		print("Saving backbone to", fileName)
		torch.save(self.backbone.state_dict(), fileName, _use_new_zipfile_serialization = False)

	def save_classifier(self, fileName):
		print("Saving classifier to", fileName)
		torch.save(self.classifier.state_dict(), fileName, _use_new_zipfile_serialization = False)

	def save_network(self, fileName):
		print("Saving BYOL network to", fileName)
		torch.save(self.state_dict(), fileName, _use_new_zipfile_serialization = False)

	def update_momentum(self, steps, total_steps):
		self.beta = 1 - (1 - self.start_beta) * (np.cos(np.pi * steps / total_steps) + 1) / 2.0

	def update_target_network(self):
		for current_params, ma_params in zip(self.backbone.parameters(), self.target_backbone.parameters()):
			old_weight, up_weight = ma_params.data, current_params.data
			ma_params.data = old_weight * self.beta + up_weight * (1 - self.beta)

		for current_params, ma_params in zip(self.encoder_projection.parameters(), self.target_projection.parameters()):
			old_weight, up_weight = ma_params.data, current_params.data
			ma_params.data = old_weight * self.beta + up_weight * (1 - self.beta)


	def encoder_forward(self, x):
		y = self.backbone(x)
		y = self.encoder_projection(y)
		y = self.encoder_prediction(y)
		return y

	def target_forward(self, x):
		y = self.target_backbone(x)
		y = self.target_projection(y)
		return y

	def forward(self, x, flag):
		if flag:
			return self.encoder_forward(x)
		else:
			return self.target_forward(x)


	def classify(self, x):
		y = self.backbone(x)
		y = self.classifier(y)
		return y

	def freeze_all_weights(self):
		for params in self.backbone.parameters():
			params.requires_grad = False
		for params in self.encoder_projection.parameters():
			params.requires_grad = False
		for params in self.encoder_prediction.parameters():
			params.requires_grad = False
		for params in self.target_backbone.parameters():
			params.requires_grad = False
		for params in self.target_projection.parameters():
			params.requires_grad = False
		for params in self.classifier.parameters():
			params.requires_grad = False

	def freeze_encoder(self):
		for params in self.backbone.parameters():
			params.requires_grad = False
		for params in self.encoder_projection.parameters():
			params.requires_grad = False
		for params in self.encoder_prediction.parameters():
			params.requires_grad = False


	def unfreeze_encoder(self):
		for params in self.backbone.parameters():
			params.requires_grad = True
		for params in self.encoder_projection.parameters():
			params.requires_grad = True
		for params in self.encoder_prediction.parameters():
			params.requires_grad = True

	def freeze_encoder_backbone(self):
		for params in self.backbone.parameters():
			params.requires_grad = False

	def unfreeze_encoder_backbone(self):
		for params in self.backbone.parameters():
			params.requires_grad = True

	def freeze_encoder_projection(self):
		for params in self.encoder_projection.parameters():
			params.requires_grad = False

	def unfreeze_encoder_projection(self):
		for params in self.encoder_projection.parameters():
			params.requires_grad = True

	def freeze_encoder_prediction(self):
		for params in self.encoder_prediction.parameters():
			params.requires_grad = False

	def unfreeze_encoder_prediction(self):
		for params in self.encoder_prediction.parameters():
			params.requires_grad = True

	def freeze_target_network(self):
		for params in self.target_backbone.parameters():
			params.requires_grad = False
		for params in self.target_projection.parameters():
			params.requires_grad = False

	def freeze_classifier(self):
		for params in self.classifier.parameters():
			params.requires_grad = False

	def unfreeze_classifier(self):
		for params in self.classifier.parameters():
			params.requires_grad = True

	def print_network_freeze(self):
		backbone_total_params = sum(p.numel() for p in self.backbone.parameters())
		backbone_trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
		encoder_projection_total_params = sum(p.numel() for p in self.encoder_projection.parameters())
		encoder_projection_trainable_params = sum(p.numel() for p in self.encoder_projection.parameters() if p.requires_grad)
		encoder_prediction_total_params = sum(p.numel() for p in self.encoder_prediction.parameters())
		encoder_prediction_trainable_params = sum(p.numel() for p in self.encoder_prediction.parameters() if p.requires_grad)
		print("{}/{} parameters in encoder are trainable.".format(backbone_trainable_params +
																  encoder_projection_trainable_params +
																  encoder_prediction_trainable_params,
																 backbone_total_params + encoder_projection_total_params
																 + encoder_prediction_total_params))

		target_backbone_total_params = sum(p.numel() for p in self.target_backbone.parameters())
		target_backbone_trainable_params = sum(p.numel() for p in self.target_backbone.parameters() if p.requires_grad)
		target_projection_total_params = sum(p.numel() for p in self.target_projection.parameters())
		target_projection_trainable_params = sum(p.numel() for p in self.target_projection.parameters() if p.requires_grad)
		print("{}/{} parameters in target are trainable.".format(target_backbone_trainable_params +
																  target_projection_trainable_params,
																 target_backbone_total_params + target_projection_total_params))
		classifier_total_params = sum(p.numel() for p in self.classifier.parameters())
		classifier_trainable_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
		print("{}/{} parameters in classifier are trainable.".format(classifier_trainable_params, classifier_total_params))


if __name__ == "__main__":
	net = BYOLNet("resnet18", True, True, 12)
	rand_tensor = torch.rand(64, 3, 224, 224)
	print(net.encoder_forward(rand_tensor).shape)
	print(net.target_forward(rand_tensor).shape)
	print(net.classify(rand_tensor).shape)
	net.print_network_freeze()
	net.freeze_encoder_backbone()
	net.freeze_target_network()
	net.print_network_freeze()
