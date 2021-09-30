import torchvision.transforms as transforms
import torch
import torch.optim as optimizers
import torch.nn as nn
import tqdm
import os
import numpy as np
import csv
import datetime
import pickle
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.tensorboard as tb
import copy

import utils
from dataloader_toybox import dataloader_toybox as data_toybox
import parser
import byol
import simclr

outputDirectory = "../output/"
mean = (0.3499, 0.4374, 0.5199)
std = (0.1623, 0.1894, 0.1775)


def learn_unsupervised(gpu, args, learner):
	rank = gpu
	dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = args['world_size'], rank = rank,)
	args['rng'], args['seed'] = set_seed(args["seed"])
	torch.cuda.set_device(gpu)
	learner.network.cuda(gpu)
	numEpochs = args['epochs1']

	color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
	gaussian_blur = transforms.GaussianBlur(kernel_size = 23)
	train_transforms = [
						# Image augmentation for first image
						transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
										transforms.RandomCrop(size = 224, padding = 25),
										transforms.RandomHorizontalFlip(p = 0.5),
										transforms.RandomApply([color_jitter], p = 0.8),
										transforms.RandomGrayscale(p = 0.2),
										#gaussian_blur,
										transforms.ToTensor(),
										transforms.Normalize(mean, std)]),
						# Image augmentation for second image
						transforms.Compose([transforms.ToPILImage(), transforms.Resize(224),
											transforms.RandomCrop(size = 224, padding = 25),
											transforms.RandomHorizontalFlip(p = 0.5),
											transforms.RandomApply([color_jitter], p = 0.8),
											transforms.RandomGrayscale(p = 0.2),
											#transforms.RandomApply([gaussian_blur], p = 0.1),
											#transforms.RandomSolarize(threshold = 0.5, p = 0.2),
											transforms.ToTensor(),
											transforms.Normalize(mean, std)]),
						]

	if gpu == 0 and args['save']:
		args['writer'] = tb.SummaryWriter(log_dir = args['writer_name'])

	if args['dataset'] == "toybox":
		trainData = data_toybox(root = "./data", rng = args["rng"], train = True, nViews = 2, size = 224,
							transform = train_transforms, fraction = args['frac1'], distort = args['distort'], adj = args['adj'],
							hyperTune = args["hypertune"])
	else:
		trainData = data_toybox(root = "./data", rng = args["rng"], train = True, nViews = 2, size = 224,
							transform = train_transforms, fraction = args['frac1'], distort = args['distort'], adj = args['adj'],
							hyperTune = args["hypertune"])

	# convert network to syncbatchnorm before wrapping with ddp
	learner.network = nn.SyncBatchNorm.convert_sync_batchnorm(learner.network)
	learner.network = nn.parallel.DistributedDataParallel(learner.network, device_ids = [gpu], broadcast_buffers = False)

	train_sampler = torch.utils.data.distributed.DistributedSampler(trainData, num_replicas = args['world_size'],
																	rank = rank, shuffle = True, seed = args['seed'])

	trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size = args['batch_size'], shuffle = False,
												  num_workers = args['workers'], sampler = train_sampler,
												  pin_memory = False, persistent_workers = True)

	unsup_train_losses = []
	learner.update_unsupervised_scheduler(num_batches = len(trainDataLoader))
	optimizer = learner.get_unsupervised_optimizer()
	scheduler = learner.get_unsupervised_scheduler()
	learner.set_linear_rampup_unsupervised_optimizer(ep = 0)
	print(len(trainDataLoader), "batches on GPU", gpu)
	for ep in range(numEpochs):
		if gpu == 0:
			tqdmBar = tqdm.tqdm(total = len(trainDataLoader))
		b = 0
		tot_loss = 0.0
		ep_losses = []
		for _, (images1, images2), _ in trainDataLoader:
			b += 1
			optimizer.zero_grad()
			loss = learner.unsupervised_loss(images1 = images1, images2 = images2, gpu = gpu)
			loss.backward()
			optimizer.step()
			dist.all_reduce(loss, dist.ReduceOp.SUM)
			loss /= args['world_size']
			tot_loss += loss
			avg_loss = tot_loss / b

			ep_losses.append(loss.item())
			if args['learner'] == 'byol':
				with torch.no_grad():
					learner.update_dual_network()

			curr_step = ep * len(trainDataLoader) + b
			if gpu == 0:
				tqdmBar.update(1)
				tqdmBar.set_description(learner.get_unsupervised_tqdm_desc(ep, numEpochs, loss = avg_loss))
				if args['save']:
					args['writer'].add_scalar("Avg_Loss/Unsupervised", avg_loss, curr_step)
					args['writer'].add_scalar("Loss/Unsupervised", loss.item(), curr_step)
					args['writer'].add_scalar("LR/Unsupervised", optimizer.param_groups[0]['lr'], curr_step)

			if ep >= args['rampup']:
				scheduler.step()
			else:
				learner.set_linear_rampup_unsupervised_optimizer(ep = curr_step)
			if args['learner'] == 'byol':
				if gpu == 0 and args['save']:
					args['writer'].add_scalar("beta", learner.get_network_beta(), curr_step)
				learner.update_beta(ep = curr_step, total_steps = numEpochs * len(trainDataLoader))

		if gpu == 0:
			tqdmBar.close()
			if args['save']:
				args['writer'].flush()
				args['writer'].close()
			unsup_train_losses.append(ep_losses)
			if (ep + 1) % args['saveRate'] == 0 and args['save'] and args['saveRate'] != -1:
				learner.save_backbone(args['saveDir'] + "unsupervised_backbone_" + str(ep + 1) + ".pt")

	if gpu == 0:
		torch.save(learner.network.module.state_dict(), "../temp.pt")
	if args["save"]:
		if gpu == 0:
			learner.save_model(args["saveDir"] + "unsupervised_final_model.pt")
			train_losses_file = args["saveDir"] + "unsupervised_train_losses.pickle"
			with open(train_losses_file, "wb") as f:
				pickle.dump(unsup_train_losses, f, protocol = pickle.DEFAULT_PROTOCOL)


def learn_supervised(gpu, args, learner, run_id = 0):
	rank = gpu
	dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = args['world_size'], rank = rank, )
	torch.cuda.set_device(gpu)
	learner.network.cuda(gpu)
	transform_train = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(padding = 10, size = 224),
										  transforms.ToTensor(), transforms.Normalize(mean, std)])
	if gpu == 0 and args['save']:
		args['writer'] = tb.SummaryWriter(args['writer_name'])

	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor(),
										 transforms.Normalize(mean, std)])

	learner.network = nn.parallel.DistributedDataParallel(learner.network, device_ids = [gpu], broadcast_buffers = False)

	trainSet = data_toybox(root = "./data", train = True, transform = [transform_train, transform_train], split = "super",
						   size = 224, fraction = args["frac2"], hyperTune = args["hypertune"], rng = args["rng"])

	testSet = data_toybox(root = "./data", train = False, transform = [transform_test, transform_test], split = "super",
						  size = 224, hyperTune = args["hypertune"], rng = args["rng"])

	train_sampler = torch.utils.data.distributed.DistributedSampler(trainSet, num_replicas = args['world_size'],
																	rank = rank)

	trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = args['batch_size'], shuffle = False, sampler =
												train_sampler, num_workers = args['workers'], pin_memory = False,
											  persistent_workers = True)

	testLoader = torch.utils.data.DataLoader(testSet, batch_size = args['batch_size'], shuffle = False,
											 pin_memory = False, num_workers = args['workers'])

	optimizer = learner.get_linear_eval_optimizer()
	numEpochs = args['epochs2']
	sup_losses = []

	for ep in range(numEpochs):
		ep_id = 0
		tot_loss = 0
		if gpu == 0:
			tqdmBar = tqdm.tqdm(total = len(trainLoader))
		ep_losses = []
		for _, images, labels in trainLoader:
			optimizer.zero_grad()
			loss = learner.linear_eval_loss(images = images, labels = labels)
			loss.backward()
			optimizer.step()

			dist.all_reduce(loss, dist.ReduceOp.SUM)
			loss /= args['world_size']
			ep_losses.append(loss.item())
			tot_loss += loss.item()
			ep_id += 1
			avg_loss = tot_loss / ep_id
			if gpu == 0:
				tqdmBar.update(1)
				tqdmBar.set_description("Epoch: {:d}/{:d} Loss: {:.4f} LR: {:.8f}".format(ep + 1, numEpochs, avg_loss,
																					  optimizer.param_groups[0]['lr']))
				if args['save']:
					args['writer'].add_scalar("Loss/Supervised", loss.item(), ep * len(trainLoader) + ep_id)
					args['writer'].add_scalar("Avg_Loss/Supervised", avg_loss, ep * len(trainLoader) + ep_id)
		if gpu == 0:
			tqdmBar.close()
			sup_losses.append(ep_losses)
		if ep % 20 == 19 and ep > 0:
			optimizer.param_groups[0]['lr'] *= 0.9

		if (ep + 1) % args['evalFreq'] == 0 and args['evalFreq'] != -1:
			if gpu == 0:
				acc, _, _ = learner.linear_acc(trainLoader = trainLoader, testLoader = testLoader, run_id = run_id)
				if args['save']:
					args['writer'].add_scalar("Accuracy", acc, ep + 1)

	print("Preparing network for evaluation. Freezing all weights....")
	learner.network.module.freeze_all_weights()
	learner.network.module.eval()
	learner.network.module.print_network_freeze()
	if gpu == 0:
		acc, _, _ = learner.linear_acc(trainLoader = trainLoader, testLoader = testLoader, run_id = run_id)
		if args['save']:
			args['writer'].add_scalar("Accuracy", acc, numEpochs)
			args['writer'].add_text("Final Accuracy", str(acc))

	if args['save']:
		if gpu == 0:
			learner.save_backbone(args['saveDir'] + 'supervised_final_backbone_' + str(run_id) + '.pt')
			learner.save_classifier(args['saveDir'] + 'supervised_final_classifier_' + str(run_id) + '.pt')
			sup_losses_file_name = args['saveDir'] + 'supervised_losses_' + str(run_id) + '.pickle'
			with open(sup_losses_file_name, "wb") as ff:
				pickle.dump(sup_losses, ff, protocol = pickle.DEFAULT_PROTOCOL)


def set_seed(sd):
	if sd == -1:
		sd = np.random.randint(0, 65536)
	print("Setting seed to", sd)
	torch.manual_seed(sd)
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = False
	rng = np.random.default_rng(sd)
	return rng, sd


def log_exp_details(tb_writer, args):
	tb_writer.add_text("hparams/lr", str(args['lr']))
	tb_writer.add_text("hparams/lr_ft", str(args['lr_ft']))
	tb_writer.add_text("hparams/e1", str(args['epochs1']))
	tb_writer.add_text("hparams/e2", str(args['epochs2']))
	tb_writer.add_text("hparams/wd", str(args['weight_decay']))
	tb_writer.add_text("exp/setting", args['distort'])
	tb_writer.add_text("hparams/seed", str(args['seed']))
	tb_writer.add_text("hparams/batch_size", str(args['batch_size']))
	tb_writer.add_text("hparams/f1", str(args["frac1"]))
	tb_writer.add_text("hparams/f1", str(args["frac2"]))
	tb_writer.add_text("exp/learner", str(args['learner']))
	tb_writer.add_text("exp/validation", str(args['hypertune']))
	if args['save']:
		tb_writer.add_text("sv/save", str(args['save']))
		tb_writer.add_text("sv/path", str(args['saveDir']))


def run_experiments(args):
	torch.autograd.set_detect_anomaly(True)
	print(torch.cuda.get_device_name(0))
	args["start"] = datetime.datetime.now()
	rng, sd = set_seed(args["seed"])
	args['seed'] = sd
	args["rng"] = rng
	args['num_nodes'] = 1
	args['num_gpus'] = torch.cuda.device_count()
	print("Splitting training over", str(torch.cuda.device_count()), "gpus.")
	args['writer_name'] = "../runs/" + args['saveName']
	args['world_size'] = args['num_gpus'] * args['num_nodes']
	if args['save']:
		log_exp_details(tb.SummaryWriter(args['writer_name']), args)
	if args['learner'] == "byol":
		learner = byol.BYOL(args = args, network = None, op_classes = 12)
	else:
		learner = simclr.SimCLR(args = args, network = None, op_classes = 12)

	learner.prepare_unsupervised()
	learner.network.train()
	os.environ['MASTER_ADDR'] = '10.20.140.47'
	os.environ['MASTER_PORT'] = '8888'
	mp.set_start_method('spawn')
	mp.spawn(learn_unsupervised, nprocs = args['num_gpus'], args = (args, learner))
	for rep in range(args['supervisedRep']):
		print("=======================================================================")
		print("Starting linear eval training run", rep + 1, "of", args['supervisedRep'])
		learner.network.load_state_dict(torch.load("../temp.pt"))
		learner.prepare_linear_eval()
		learner.network.eval()
		mp.spawn(learn_supervised, nprocs = args['num_gpus'], args = (args, learner, rep))
		print("=======================================================================")


if __name__ == "__main__":
	exp_args = vars(parser.get_parser("SSL Parser"))
	if exp_args['save']:
		try:
			assert exp_args['saveName'] != ""
		except AssertionError:
			raise RuntimeError("Please provide name of directory in which output files will be stored.")
		if not os.path.isdir(outputDirectory):
			os.mkdir(outputDirectory)
		exp_args["saveDir"] = outputDirectory + exp_args['saveName'] + "/"
		try:
			assert not os.path.isdir(exp_args["saveDir"])
		except AssertionError:
			raise RuntimeError("Provided output directory already exists. Please provide new name.")
		os.mkdir(exp_args["saveDir"])

	run_experiments(args = exp_args)
