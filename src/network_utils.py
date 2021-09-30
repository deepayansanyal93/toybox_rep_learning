import torch.nn as nn


class Identity(nn.Module):
	def __init__(self):
		super().__init__()

	def __call__(self, x):
		return x

	def __repr__(self):
		return "Identity()"


def init_weights(m):
	if type(m) == nn.Linear or type(m) == nn.Conv2d:
		nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			m.bias.data.fill_(0.01)
