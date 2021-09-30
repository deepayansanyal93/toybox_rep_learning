import torchvision.transforms as transforms
import torch.nn.functional as F
import torch

mean = (0.3499, 0.4374, 0.5199)
std = (0.1623, 0.1894, 0.1775)


class UnNormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		Returns:
			Tensor: Normalized image.
		"""
		for t, m, s in zip(tensor, self.mean, self.std):
			t.mul_(s).add_(m)
			# The normalize code -> t.sub_(m).div_(s)
		return tensor


def get_train_transform(tr):
	s = 1
	color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
	if tr == 1:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
										transforms.RandomCrop(size = 224, padding = 25),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.RandomApply([color_jitter], p = 0.8),
										transforms.RandomGrayscale(p = 0.2),
										transforms.GaussianBlur(kernel_size = 23),
										transforms.RandomSolarize(threshold = 0.5, p = 0.2),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])
	elif tr == 2:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
										transforms.RandomCrop(size = 224, padding = 25),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.RandomApply([color_jitter], p = 0.8),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])

	elif tr == 3:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
										transforms.RandomCrop(size = 224, padding = 25),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.ToTensor(),
										transforms.Normalize(mean, std)])
	elif tr == 4:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
							transforms.RandomHorizontalFlip(p = 0.5),
							transforms.ToTensor(),
							transforms.Normalize(mean, std)])
	else:
		transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
							transforms.RandomHorizontalFlip(p = 0.5),
							transforms.RandomApply([color_jitter], p = 0.8),
							transforms.ToTensor(),
							transforms.Normalize(mean, std)])

	return transform


def loss_fn(x, y):
	x = F.normalize(x, dim=-1, p=2)
	y = F.normalize(y, dim=-1, p=2)
	loss = torch.tensor(2) - 2 * (x * y).sum(dim=-1)
	return loss


def info_nce_loss(features):
	batchSize = features.shape[0] / 2
	labels = torch.cat([torch.arange(batchSize) for _ in range(2)], dim = 0)
	labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
	labels = labels.cuda()

	features = F.normalize(features, dim = 1)

	similarity_matrix = torch.matmul(features, torch.transpose(features, 0, 1))
	# discard the main diagonal from both: labels and similarities matrix
	mask = torch.eye(labels.shape[0], dtype = torch.bool).cuda()
	labels = labels[~mask].view(labels.shape[0], -1)
	similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
	positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
	negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

	logits = torch.cat([positives, negatives], dim = 1)
	labels = torch.zeros(logits.shape[0], dtype = torch.long).cuda()

	logits = logits / 0.5
	return logits, labels


def calc_accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batchSize = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		target_reshaped = torch.reshape(target, (1, -1)).repeat(maxk, 1)
		correct_top_k = torch.eq(pred, target_reshaped)
		pred_1 = pred[0]
		res = []
		for k in topk:
			correct_k = correct_top_k[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(torch.mul(correct_k, 100.0 / batchSize))
		return res, pred_1


def get_dist_mat(act_a, act_b):
	aR = torch.repeat_interleave(act_a, act_b.shape[0], dim = 0)
	bR = act_b.repeat(act_a.shape[0], 1)
	dist_mat = torch.sqrt(torch.pow(aR - bR, 2).sum(dim = 1))
	dist_mat = dist_mat.view(act_a.shape[0], -1)
	return dist_mat


def get_dist(act_a, act_b):
	distMat = None
	for i in range(act_a.shape[0]):
		dists = torch.sqrt(torch.pow(act_a[i] - act_b, 2).sum(dim = 1)).unsqueeze(0)
		if distMat is None:
			distMat = dists
		else:
			distMat = torch.cat((distMat, dists), dim = 0)

	return distMat


def knn_eval(network, trainData, testData):
	trainActs = None
	trainLabels = None
	for _, (trainIms, _), labels in trainData:
		with torch.no_grad():
			activations = network.encoder_backbone(trainIms.cuda())
			if trainActs is None:
				trainActs = activations
				trainLabels = labels
			else:
				trainActs = torch.cat((trainActs, activations))
				trainLabels = torch.cat((trainLabels, labels))
	testActs = None
	testLabels = None
	i = 0
	for _, (_, testIms, labels) in enumerate(testData):
		with torch.no_grad():
			activations = network.encoder_backbone(testIms.cuda())
			i += 1
			if testActs is None:
				testActs = activations
				testLabels = labels
			else:
				testActs = torch.cat((testActs, activations))
				testLabels = torch.cat((testLabels, labels))

	dist_matrix = get_dist(act_a = testActs.cuda(), act_b = trainActs.cuda())
	topkDist, topkInd = torch.topk(dist_matrix, k = 200, dim = 1, largest = False)
	preds = trainLabels[topkInd.squeeze()]
	predsMode, _ = torch.mode(preds, dim = 1)
	acc = 100 * torch.eq(predsMode, testLabels).float().sum()/testLabels.shape[0]
	return acc.numpy()
