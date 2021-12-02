import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import ensembling.model as model
import ensembling.config as config
import ensembling.evaluate as evaluate
import ensembling.data_utils as data_utils


parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=20,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
parser.add_argument("--classification",
	type=bool,
	default=False,
	help="Whether to use classification or not")
parser.add_argument("--num_ensemble",
	type=int,
	default=5,
	help="Number of ensemble members to creates")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

print("Args.classificaiton ", args.classification)
############################## PREPARE DATASET ##########################
if not args.classification:
	print("In regression")
	train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all_regression_lastfm()
else:
	train_data, test_data, user_num ,item_num, train_mat, test_mat, train_labels, test_labels = data_utils.load_all_classification_lastfm()

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_labels, train_mat, 1, True, args.classification)
train_dataset.ng_sample()
print("Making test loader")
#print("Test data is ", test_data)
print("Length of test data before test loader is ", len(test_data))
test_dataset = data_utils.NCFData(
		test_data, item_num, test_labels, test_mat, 0, False, args.classification)
print("test dataset is false for training? ", test_dataset.is_training)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=2)

########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path)
	MLP_model = torch.load(config.MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

ensemble = [model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, config.model, args.classification, GMF_model, MLP_model).cuda() for _ in range(args.num_ensemble)]

if not args.classification:
	loss_functions = [nn.BCEWithLogitsLoss() for _ in range(args.num_ensemble)]
else:
	loss_functions = [nn.CrossEntropyLoss() for _ in range(args.num_ensemble)]

if config.model == 'NeuMF-pre':
	optimizers = [optim.SGD(m.parameters(), lr=args.lr) for m in ensemble]
else:
	optimizers = [optim.Adam(m.parameters(), lr=args.lr) for m in ensemble]

# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################
count, best_hr = 0, 0
best_acc = 0
for epoch in range(args.epochs):
	for m in ensemble:
		m.train() # Enable dropout (if have).
	start_time = time.time()
	# train_loader.dataset.ng_sample()
	
	for model_index in range(len(ensemble)):
		m = ensemble[model_index]
		for user, item, label in train_loader:
			user = user.cuda()
			item = item.cuda()
			if not args.classification:
				label = label.float().cuda()
			else:
				label = label.cuda()

			m.zero_grad()
			prediction = m(user, item)
			# print("prediction is ", prediction)
			# print("label is ", label)
			loss = loss_functions[model_index](prediction, label)
			loss.backward()
			optimizers[model_index].step()
			# writer.add_scalar('data/loss', loss.item(), count)
			count += 1

	for m in ensemble:
		m.eval()

	if not args.classification:
		HR, NDCG = evaluate.metrics(model, test_loader, args.top_k, args.classification)

		elapsed_time = time.time() - start_time
		print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

		if HR > best_hr:
			best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
			if args.out:
				if not os.path.exists(config.model_path):
					os.mkdir(config.model_path)
				torch.save(model, 
					'{}{}.pth'.format(config.model_path, config.model))

		print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
									best_epoch, best_hr, best_ndcg))
	else:
		accuracy, uncertainty = evaluate.uncertainty_and_accuracy(ensemble, test_loader)
		elapsed_time = time.time() - start_time
		print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		print("Accuracy", accuracy, "Uncertainty" , uncertainty)
		if accuracy > best_acc:
			best_acc, best_epoch = accuracy, epoch
			if args.out:
				if not os.path.exists(config.model_path):
					os.mkdir(config.model_path)
				# # torch.save(model, 
				# 	'{}{}.pth'.format(config.model_path, config.model))

		print("End. Best epoch {:03d}: Accuracy = {:.3f}".format(
									best_epoch, best_acc))
