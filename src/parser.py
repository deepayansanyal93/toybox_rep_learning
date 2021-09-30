import argparse


def get_parser(desc):
	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument('--learner', '-ssl', required = True, choices = ['byol', 'simclr'])
	parser.add_argument('--batch-size', '-b', default = 64, type = int, help = "Batch Size for experiments")
	parser.add_argument('--epochs1', '-e1', default = 80, type = int, help = "Number of epochs for unsupervised training")
	parser.add_argument('--epochs2', '-e2', default = 40, type = int, help = "Number of epochs of supervised training")
	parser.add_argument('--resume', '-r', default = False, action = 'store_true', help = "Ignore if training from scratch"
																						 ", use if continuing training")
	parser.add_argument('--resumeFile', '-rf', default = "", type = str)
	parser.add_argument('--seed', '-s', default = -1, type = int, help = "Seed for training")
	parser.add_argument('--distort', '-d', choices = ['self', 'object', 'transform', 'class'], help = "Choose "
								"distortion for images. self for augmentations, object for other view of same object, "
								"transform for other image of object from same video, class for image from any other"
								"frame of same class.")
	parser.add_argument('--lr', '-lr', default = 0.16, type = float, help = "Learning rate for unsupervised learning")
	parser.add_argument('--lr-ft', '-lr_ft', default = 0.02, type = float, help = "Learning rate for supervised "
																				   "training.")
	parser.add_argument('--save', '-sv', default = False, action = "store_true", help = "Use to save trained models.")
	parser.add_argument('--adj', '-a', default = -1, type = int)
	parser.add_argument('--frac1', '-f1', default = 1.0, type = float)
	parser.add_argument('--frac2', '-f2', default = 0.1, type = float)
	parser.add_argument('--freeze-backbone', '-fb', default = True, action = 'store_false')
	parser.add_argument('--hypertune', '-ht', default = False, action = 'store_true')
	parser.add_argument('--saveName', '-sn', default = "", type = str)
	parser.add_argument('--saveRate', '-sr', default = -1, type = int)
	parser.add_argument('--transform', '-tr', default = 1, type = int)
	parser.add_argument('--temperature', '-t', default = 0.5, type = float)
	parser.add_argument('--weight-decay', '-wd', default = 1e-6, type = float)
	parser.add_argument('--epochsRan', '-er', default = -1, type = int)
	parser.add_argument('--supervisedRep', '-rep', default = -1, type = int)
	parser.add_argument('--dataset', '-data', required = True, type = str)
	parser.add_argument('--sessionSplit', '-sess', default = False, action = 'store_true')
	parser.add_argument('--distortArg', '-da', default = False, action = 'store_true')
	parser.add_argument('--margin', '-m', default = 1.0, type = float)
	parser.add_argument('--loss', '-l', default = 'cosine', type = str)
	parser.add_argument('--workers', '-w', default = 4, type = int)
	parser.add_argument('--ratio', '-rt', default = 0.5, type = float)
	parser.add_argument('--gap', '-g', default = 2, type = int)
	parser.add_argument('--interpolate', '-i', default = False, action = 'store_true')
	parser.add_argument('--transfer', default = False, action = 'store_true')
	parser.add_argument('--rampup', '-ramp', default = 10, type = int)
	parser.add_argument('--evalFreq', '-ef', default = -1, type = int)

	return parser.parse_args()
