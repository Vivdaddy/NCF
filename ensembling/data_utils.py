import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data
from torch.nn.functional import one_hot
import torch
import ensembling.config as config


def load_all(test_num=100):
	""" We load all the three file here to save time in each epoch. """
	train_data = pd.read_csv(
		config.train_rating, 
		sep='\t', header=None, names=['user', 'item'], 
		usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

	user_num = train_data['user'].max() + 1
	item_num = train_data['item'].max() + 1

	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0

	test_data = []
	with open(config.test_negative, 'r') as fd:
		line = fd.readline()
		while line != None and line != '':
			arr = line.split('\t')
			u = eval(arr[0])[0]
			test_data.append([u, eval(arr[0])[1]])
			for i in arr[1:]:
				test_data.append([u, int(i)])
			line = fd.readline()
	# print(test_data[:3])
	return train_data, test_data, user_num, item_num, train_mat



def load_all_classification(test_num=100):
	""" We load all the three file here to save time in each epoch. """
	train_data = pd.read_csv(
		config.train_rating, 
		sep='\t', header=None, names=['user', 'item', 'rating'], 
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})

	user_num = train_data['user'].max() + 1
	item_num = train_data['item'].max() + 1

	train_data = train_data.values.tolist()
	train_labels = []
	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = x[2]
		train_labels.append(x[2])

	test_data = []
	test_df = pd.read_csv(
		config.test_rating, 
		sep='\t', header=None, names=['user', 'item', 'rating'], 
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})

	test_data = test_df.values.tolist()
	test_labels = []
	# load ratings as a dok matrix
	test_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in test_data:
		test_mat[x[0], x[1]] = x[2]
		test_labels.append(x[2])
	# print("test_df is \n", test_df.head())

	# with open(config.test_negative, 'r') as fd:
	# 	line = fd.readline()
	# 	while line != None and line != '':
	# 		arr = line.split('\t')
	# 		u = eval(arr[0])[0]
	# 		# print("u is ", u)
	# 		# print("For user u \n", test_df[test_df['user']==u])
	# 		test_data.append([u, eval(arr[0])[1], test_df[test_df['user'] == u]['rating']])
	# 		for i in arr[1:]:
	# 			test_data.append([u, int(i), 0])
	# 		line = fd.readline()
	# print("test data is \n", test_data[:3])
	return train_data, test_data, user_num, item_num, train_mat, train_labels, test_labels



def load_all_classification_lastfm(test_num=100):
	""" We load all the three file here to save time in each epoch. """
	train_data = pd.read_csv(
		config.train_rating, 
		sep='\t', header=None, names=['user', 'item', 'timestamps'], 
		usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})
	train_data = train_data.sort_values('timestamps')
	length = len(train_data.index)
	train_length = (9 * length) // 10
	test_length = length - train_length
	train_data = train_data.values[:, -3:]
	unique_users = sorted(list(set(data[:, 0])))
	unique_items = sorted(list(set(data[:, 1])))
	user_dic = {user:idx for (idx,user) in enumerate(unique_users)}
	item_dic = {item:idx for (idx,item) in enumerate(unique_items)}
	user_num = len(unique_users) + 1
	item_num = len(unique_items) + 1
	for (idx, row) in enumerate(train_data):
		user,item,time = user_dic[row[0]],item_dic[row[1]],row[2]
		train_data[idx,0],train_data[idx,1] = int(user),int(item)
	train_data = train_data.values.tolist()
	train_labels = []
	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data.head(train_length):
		train_mat[x[0], x[1]] = 1.0
		train_labels.append(1)

	test_data = train_data[-test_length -1:]
	train_data = train_data[:train_length]
	test_labels = []
	# load ratings as a dok matrix
	test_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in test_data:
		test_mat[x[0], x[1]] = 1
		test_labels.append(1)
	# print("test_df is \n", test_df.head())

	# with open(config.test_negative, 'r') as fd:
	# 	line = fd.readline()
	# 	while line != None and line != '':
	# 		arr = line.split('\t')
	# 		u = eval(arr[0])[0]
	# 		# print("u is ", u)
	# 		# print("For user u \n", test_df[test_df['user']==u])
	# 		test_data.append([u, eval(arr[0])[1], test_df[test_df['user'] == u]['rating']])
	# 		for i in arr[1:]:
	# 			test_data.append([u, int(i), 0])
	# 		line = fd.readline()
	# print("test data is \n", test_data[:3])
	return train_data, test_data, user_num, item_num, train_mat, train_labels, test_labels


class NCFData(data.Dataset):
	def __init__(self, features, 
				num_item, labels, train_mat=None, num_ng=0, is_training=None, classification=True):
		super(NCFData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features_ps = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training
		self.labels = labels
		self.classification = classification

	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'
		if not self.classification:
			self.features_ng = []
			for x in self.features_ps:
				u = x[0]
				for t in range(self.num_ng):
					j = np.random.randint(self.num_item)
					while (u, j) in self.train_mat:
						j = np.random.randint(self.num_item)
					self.features_ng.append([u, j])

			labels_ps = [1 for _ in range(len(self.features_ps))]
			labels_ng = [0 for _ in range(len(self.features_ng))]

			self.features_fill = self.features_ps + self.features_ng
			self.labels_fill = labels_ps + labels_ng
		else:
			# print(self.features_ps)
			self.features_ng = []
			labels_ps = []
			for x in self.features_ps:
				u = x[0]
				for t in range(self.num_ng):
					j = np.random.randint(self.num_item)
					while (u, j) in self.train_mat:
						j = np.random.randint(self.num_item)
					self.features_ng.append([u, j])
					# labels_ps.append(x[2])

			labels_ps = [i[2] for i in self.features_ps]
			labels_ng = [0 for _ in range(len(self.features_ng))]

			self.features_fill = self.features_ps +self.features_ng
			# self.features_fill = self.features_ps
			labels =  labels_ps + labels_ng
			# labels = labels_ps
			####
			# self.labels_fill = one_hot(torch.Tensor(labels).to(torch.long), num_classes=6)
			self.labels_fill = torch.Tensor(labels).to(torch.long)
	


	def __len__(self):
		return (self.num_ng + 1) * len(self.labels)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training \
					else self.features_ps
		labels = self.labels_fill if self.is_training \
					else self.labels
		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]
		return user, item ,label
		