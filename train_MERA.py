#!/usr/bin/env python3
import time
import torch
from models.lotenet import loTeNet, ConvTeNet
from models.MERA import MERAnet_clean as MERAnet

from torchvision import transforms, datasets
import pdb
from utils.lidc_dataset import LIDC
from utils.tools import *
from models.Densenet import *
from models.Densenet import DenseNet, FrDenseNet
import argparse

# Globally load device identifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global global_bs
def evaluate(loader):
	### Evaluation funcntion for validation/testing

	with torch.no_grad():
		vl_acc = 0.
		vl_loss = 0.
		#labelsNp = np.zeros(1)
		#predsNp = np.zeros(1)
		model.eval()

		for i, (inputs, labels) in enumerate(loader):

			inputs = inputs.to(device)
			labels = labels.to(device)
			#labelsNp = np.concatenate((labelsNp, labels.cpu().numpy()))

			# Inference
			#scores = torch.sigmoid(model(inputs))
			sm = torch.nn.Softmax(dim=1)
			scores = sm(model(inputs))

			scores = scores.float()
			labels = labels.long()
			preds = scores

			loss = loss_fun(scores, labels)
			#predsNp = np.concatenate((predsNp, preds.cpu().numpy()))
			vl_loss += loss.item()

		# Compute AUC over the full (valid/test) set
		vl_acc = computeAuc(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
		vl_loss = vl_loss/len(loader)

	return vl_acc, vl_loss

# Miscellaneous initialization
torch.manual_seed(1)
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--l2', type=float, default=0, help='L2 regularisation')
parser.add_argument('--aug', action='store_true', default=False, help='Use data augmentation')
parser.add_argument('--data_path', type=str, default='./data/',help='Path to data.')
parser.add_argument('--bond_dim', type=int, default=5, help='MPS Bond dimension')
parser.add_argument('--nChannel', type=int, default=1, help='Number of input channels')
parser.add_argument('--dense_net', action='store_true',
					default=False, help='Using Dense Net model')
parser.add_argument('--MERA', action='store_true',
					default=False, help='Using Conv style Tensor Net model')
parser.add_argument('--kernel', nargs='+', type=int)

parser.add_argument(
        "--gpu",
        default=0,
        help="the directory of the model"
    )
args = parser.parse_args()
global_bs = args.batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:'+str(args.gpu))
print(device)
with torch.cuda.device('cuda:'+str(args.gpu)):
	batch_size = args.batch_size

	# LoTeNet parameters
	adaptive_mode = False
	periodic_bc   = False

	kernel = args.kernel  # Stride along spatial dimensions
	output_dim = 2 # output dimension

	feature_dim = 2

	logFile = time.strftime("%Y%m%d_%H_%M")+'.txt'
	makeLogFile(logFile)

	normTensor = 0.5*torch.ones(args.nChannel)
	### Data processing and loading....
	trans_valid = transforms.Compose([transforms.Normalize(mean=normTensor,std=normTensor)])

	if args.aug:
		trans_train = transforms.Compose([transforms.ToPILImage(),
							  transforms.RandomHorizontalFlip(),
							  transforms.RandomVerticalFlip(),
							  transforms.RandomRotation(20),
							  transforms.ToTensor(),
							  transforms.Normalize(mean=normTensor,std=normTensor)])
		print("Using Augmentation....")
	else:
		trans_train = trans_valid
		print("No augmentation....")

	# Load processed LIDC data
	dataset_train = LIDC(split='Train', data_dir=args.data_path,
						transform=trans_train,rater=4)
	dataset_valid = LIDC(split='Valid', data_dir=args.data_path,
						transform=trans_valid,rater=4)
	dataset_test = LIDC(split='Test', data_dir=args.data_path,
						transform=trans_valid,rater=4)

	num_train = len(dataset_train)
	num_valid = len(dataset_valid)
	num_test = len(dataset_test)
	print("Num. train = %d, Num. val = %d"%(num_train,num_valid))

	loader_train = DataLoader(dataset = dataset_train, drop_last=True,
							  batch_size=batch_size, shuffle=True)
	loader_valid = DataLoader(dataset = dataset_valid, drop_last=True,
							  batch_size=batch_size, shuffle=False)
	loader_test = DataLoader(dataset = dataset_test, drop_last=True,
							 batch_size=batch_size, shuffle=False)

	# Initiliaze input dimensions
	dim = torch.ShortTensor(list(dataset_train[0][0].shape[1:]))
	nCh = int(dataset_train[0][0].shape[0])

	# Initialize the models
	if not args.dense_net:
		#print(args.convTN)
		if args.MERA:
			print("Using MERA")
			model = MERAnet(input_dim=dim, output_dim=output_dim,
							nCh=nCh, kernel=kernel,
							bond_dim=args.bond_dim, feature_dim=feature_dim,
							adaptive_mode=adaptive_mode, periodic_bc=periodic_bc, virtual_dim=1)
		else:
			print("Using LoTeNet")
			model = loTeNet(input_dim=dim, output_dim=output_dim,
						  nCh=nCh, kernel=kernel,
						  bond_dim=args.bond_dim, feature_dim=feature_dim,
						  adaptive_mode=adaptive_mode, periodic_bc=periodic_bc, virtual_dim=1)
	else:
		print("Densenet Baseline!")
		model = FrDenseNet(depth=40, growthRate=12,
						reduction=0.5,bottleneck=True,nClasses=output_dim)

	# Choose loss function and optimizer
	#loss_fun = torch.nn.BCELoss()

	loss_fun = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
								 weight_decay=args.l2)

	nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Number of parameters:%d"%(nParam))
	print("Maximum MPS bond dimension = {args.bond_dim}")
	with open(logFile, "a") as f:
		print("Bond dim: %d"%(args.bond_dim))
		print("Number of parameters:%d"%(nParam))

	print("Using Adam w/ learning rate = {args.lr:.1e}")
	print("Feature_dim: %d, nCh: %d, B:%d"%(feature_dim,nCh,batch_size))

	model = model.to(device)
	nValid = len(loader_valid)
	nTrain = len(loader_train)
	nTest = len(loader_test)

	maxAuc = 0
	minLoss = 1e3
	convCheck = 5
	convIter = 0

	# Let's start training!
	for epoch in range(args.num_epochs):
		running_loss = 0.
		running_acc = 0.
		t = time.time()
		model.train()
		#predsNp = np.zeros(1)
		#labelsNp = np.zeros(1)

		for i, (inputs, labels) in enumerate(loader_train):

			inputs = inputs.to(device)
			labels = labels.to(device)
			#labelsNp = np.concatenate((labelsNp, labels.cpu().numpy()))
			sm = torch.nn.Softmax(dim=1)
			scores = sm(model(inputs))
			#scores = torch.sigmoid(model(inputs))
			#labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=output_dim)
			preds = scores
			preds = preds.float()
			#print(scores.shape)
			# print(labels.shape)
			# print(labels.detach().cpu().numpy())
			# assert False
			loss = loss_fun(scores, labels.long())

			with torch.no_grad():
				#predsNp = np.concatenate((predsNp, preds.detach().cpu().numpy()))
				running_loss += loss

			# Backpropagate and update parameters
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (i+1) % 5 == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
					   .format(epoch+1, args.num_epochs, i+1, nTrain, loss.item()))

		accuracy = computeAuc(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())

		# Evaluate on Validation set
		with torch.no_grad():

			vl_acc, vl_loss = evaluate(loader_valid)
			if vl_acc > maxAuc or vl_loss < minLoss:
				if vl_loss < minLoss:
					minLoss = vl_loss
				if vl_acc > maxAuc:
					### Predict on test set
					ts_acc, ts_loss = evaluate(loader_test)
					maxAuc = vl_acc
					print('New Max: %.4f'%maxAuc)
					print('Test Set Loss:%.4f	Auc:%.4f'%(ts_loss, ts_acc))
					with open(logFile,"a") as f:
						print('Test Set Loss:%.4f	Auc:%.4f'%(ts_loss, ts_acc))
				convEpoch = epoch
				convIter = 0
			else:
				convIter += 1
			if convIter == convCheck:
				if not args.dense_net:
					print("MPS")
				else:
					print("DenseNet")
				print("Converged at epoch:%d with AUC:%.4f"%(convEpoch+1,maxAuc))

				#break
		writeLog(logFile, epoch, running_loss/nTrain, accuracy,
				vl_loss, vl_acc, time.time()-t)
