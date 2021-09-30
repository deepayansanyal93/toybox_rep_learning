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


def trial_method(gpu, args):
	rank = gpu
	dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = args['world_size'], rank = rank, )
	torch.cuda.set_device(gpu)
	x = torch.from_numpy(np.random.randint(0, 10, 1))
	x = x.cuda(gpu)
	y = torch.add(x, torch.tensor([1.0]).cuda(gpu))
	print(gpu, x, y)
	z = dist.all_reduce(y, torch.distributed.ReduceOp.SUM)
	print(gpu, z)


if __name__ == "__main__":
	byol_args = {'num_nodes': 1, 'num_gpus': 1}
	byol_args['world_size'] = byol_args['num_gpus'] * byol_args['num_nodes']
	os.environ['MASTER_ADDR'] = '10.20.140.47'
	os.environ['MASTER_PORT'] = '8888'
	mp.set_start_method('spawn')
	mp.spawn(trial_method, nprocs = byol_args['num_gpus'], args = (byol_args, ))
