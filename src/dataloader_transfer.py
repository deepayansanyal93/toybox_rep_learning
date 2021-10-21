from torchvision.datasets import CIFAR10
import numpy as np


class fCIFAR10(CIFAR10):

	def __init__(self, root, train, download, transform, rng, hypertune = True, frac = 1.0):
		if hypertune:
			super(fCIFAR10, self).__init__(root = root, train = True, download = download)
		else:
			super(fCIFAR10, self).__init__(root = root, train = train, download = download)
		self.train = train
		self.transform = transform
		self.frac = frac
		self.hypertune = hypertune
		self.rng = rng

		if self.hypertune:
			if self.train:
				range_low = 0
				range_high = int(0.8 * len(self.data))
			else:
				range_low = int(0.8 * len(self.data))
				range_high = len(self.data)
		else:
			range_low = 0
			range_high = len(self.data)

		arr = np.arange(range_low, range_high)
		print("Split:", self.train, np.min(arr), np.max(arr))
		len_data = range_high - range_low

		indices = self.rng.choice(arr,  size = int(frac * len_data), replace = False)

		unique = len(indices) == len(set(indices))
		assert unique
		assert len(indices) == int(frac * len_data)

		if self.train:
			self.train_data = self.data[indices]
			self.train_labels = np.array(self.targets)[indices]
		else:
			self.test_data = self.data[indices]
			self.test_labels = np.array(self.targets)[indices]


	def __len__(self):
		if self.train:
			return len(self.train_data)
		else:
			return len(self.test_labels)

	def __getitem__(self, item):
		if self.train:
			img = self.train_data[item]
			target = self.train_labels[item]
		else:
			img = self.test_data[item]
			target = self.test_labels[item]
		img = self.transform(img)
		return item, img, target
