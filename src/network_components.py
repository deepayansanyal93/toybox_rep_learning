import torchvision.models as models
import torch.nn as nn
import torch

import network_utils as nutils


class netBackbone(nn.Module):
	"""Backbone architectures for different network models.
	"""
	models_supported = {
		"resnet18": models.resnet18,
		"resnet34": models.resnet34,
		"resnet50": models.resnet50,
		}

	def __init__(self, backbone: str, model_name: str):
		super().__init__()
		self.backbone_name = backbone.lower()
		self.model_name = model_name
		try:
			assert self.backbone_name in self.models_supported.keys()
		except AssertionError:
			raise AssertionError("backbone architecture not supported.")
		self.network = self.models_supported[self.backbone_name](pretrained = False, num_classes = 256)
		self.__num_feat = self.network.fc.in_features
		self.network.fc = nutils.Identity()


	def __str__(self) -> str:
		return '{} backbone'.format(self.model_name)


	def forward(self, x):
		return self.network(x)


	def freeze_params(self) -> None:
		print("Freezing {}".format(self))
		for name, param in self.network.named_parameters():
			param.requires_grad = False
		self.network.eval()

	def unfreeze_params(self) -> None:
		print("Unfreezing {}".format(self))
		for name, param in self.network.named_parameters():
			param.requires_grad = True
		self.network.train()

	def get_feat_size(self) -> int:
		return self.__num_feat


class MLP(nn.Module):
	"""Generic Class for Multilayer Perceptrons
	"""

	def __init__(self, num_neurons, relu_in_middle, bn_in_middle = False, model_name = "Generic MLP"):
		super().__init__()
		self.num_neurons = num_neurons
		self.relu_in_middle = relu_in_middle
		self.bn_in_middle = bn_in_middle
		self.model_name = model_name
		self.num_hidden_layers = len(self.num_neurons) - 2
		try:
			assert self.num_hidden_layers == 1
		except AssertionError:
			print("MLPs with one hidden layer supported only.")

		self.network = nn.Sequential()
		for i in range(self.num_hidden_layers + 1):
			self.network.add_module(self.model_name + "_linear_" + str(i + 1), nn.Linear(self.num_neurons[i],
																						 self.num_neurons[i + 1]))
			if i != self.num_hidden_layers and self.bn_in_middle is True:
				self.network.add_module(self.model_name + "_bn_" + str(i + 1), nn.BatchNorm1d(
					num_features = self.num_neurons[i + 1]))
			if i != self.num_hidden_layers and self.relu_in_middle is True:
				self.network.add_module(self.model_name + "_relu_" + str(i + 1), nn.ReLU())

	def __str__(self):
		return self.model_name


	def forward(self, x):
		return self.network(x)


	def freeze_params(self):
		print("Freezing MLP {}".format(self))
		for name, param in self.network.named_parameters():
			param.requires_grad = False
		self.network.eval()


	def unfreeze_params(self):
		print("Unfreezing MLP {}".format(self))
		for name, param in self.network.named_parameters():
			param.requires_grad = False
		self.network.train()


if __name__ == "__main__":
	nb = netBackbone(backbone = "resnet18", model_name = "resnet18 backbone")
	print(nb.get_feat_size())
	feat_size = nb.get_feat_size()
	mlp1 = MLP(num_neurons = [feat_size, 10], relu_in_middle = True, model_name = "Projection Head")
	print(str(mlp1), repr(mlp1))
	mlp1.freeze_params()
	mlp1.unfreeze_params()

	rand_tensor = torch.rand(64, 3, 224, 224)
	b_output = nb(rand_tensor)
	print(b_output.shape)
	m_output = mlp1(b_output)
	print(m_output.shape)
