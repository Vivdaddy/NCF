# In[0]
import torch
from torch.functional import _return_counts
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import os
#import rbo
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

from en_lstm_MLP_split import Ensemble_LSTM

# In[1] Experiment Setting & Load Data

"""\
Description:
------------
    Ensemble LSTM experiment setting

Parameters:
-----------
    input_path: input dataset directory
    data_kind: dataset - lastfm, lastfm_ori, foursquare, wikipedia, reddit 
    output_path: output result directory

    look_back: size of user trajectory for train
    epochs: maximum epochs for Ensemble LSTM
    pert: size of random deletion for bootstrapping
"""

input_path = "/home/vanand37/Research/NCF/aggregation/data/"
data_kind = "lastfm"
output_path = data_kind + "/user_MLP_split/"

look_back = 50
epochs = 50 # max epochs for each LSTM & MLP
num_ensemble = 10
# drop_rate = 0
# pert_ratios = [0] # without bootstrapping

# MLP parameter setting
mlp_initial = False
mlp_bias = True

# In[2] Experiment Setting & Load Data

def train_test_split_user(data=[], test_ratio = 0.1, valid = None):
    
    (users,counts) = np.unique(data[:,0],return_counts = True)
    users = users[counts>=10]

    user_dic = {int(user):idx for (idx,user) in enumerate(users)}
    new_data = []
    for i in range(data.shape[0]):
        if int(data[i,0]) in user_dic:
            new_data.append([int(data[i,0]),int(data[i,1]),data[i,2],0])

    new_data = np.array(new_data)
    new_data = new_data[np.argsort(new_data[:,2])]

    sequence_dic = {int(user):[] for user in set(data[:,0])}

    for i in range(new_data.shape[0]):
        sequence_dic[int(new_data[i,0])].append([i,int(new_data[i,1]),new_data[i,2]])
    
    for user in sequence_dic.keys():
        cur_test = int(test_ratio * len(sequence_dic[user]))
        for i in range(cur_test):
            interaction = sequence_dic[user].pop()
            new_data[interaction[0],3] = 2
        
        if valid != None:
            cur_valid = int(test_ratio * len(sequence_dic[user]))
            for i in range(cur_valid):
                interaction = sequence_dic[user].pop()
                new_data[interaction[0],3] = 1

    return new_data


def sequence_generator(data, look_back = 50):

    train, test, valid = [],[],[]
    unique_users = set(data[:,0])
    items_per_user = {int(user):[0 for i in range(look_back)] for user in unique_users}
    
    for (idx,row) in enumerate(data):
      user,item,time = int(row[0]),int(row[1]),row[2]
      items_per_user[user] = items_per_user[user][1:]+[item+1]
      current_items = items_per_user[user]
      if row[3]==0:
        train.append([current_items[:-1],current_items[-1]])
      elif row[3]==2:
        test.append([current_items[:-1],current_items[-1]])
      else:
        valid.append([current_items[:-1],current_items[-1]])
                                          
    return train, valid, test


print("Loading data: ", data_kind)
raw_data = pd.read_csv(input_path + data_kind +".tsv", sep='\t', header=None) 
data = raw_data.values[:,-3:]

# reduce total data size: pct * 100 %
pct = 1
data = data[:int(len(data)*pct)]

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is ", device)

# Train - Valid - Test split wrt individual user trajectory
original_data = train_test_split_user(data=data, test_ratio = 0.1, valid = "ON")

unique_users = sorted(list(set(original_data[:, 0])))
unique_items = sorted(list(set(original_data[:, 1])))
user_dic = {user:idx for (idx,user) in enumerate(unique_users)}
item_dic = {item:idx for (idx,item) in enumerate(unique_items)}

# Sort user ID, item ID in ascending order from 0 to n
for (idx, row) in enumerate(original_data):
    user,item,time = user_dic[row[0]],item_dic[row[1]],row[2]
    original_data[idx,0],original_data[idx,1] = int(user),int(item)

print("unique users = {}, unique items = {}, user/item ratio = {:.1f}".format(len(unique_users),len(unique_items),len(unique_users)/len(unique_items)))

# User Interation [user, item, time] to Item Sequence [item sequence, next ground-truth label]
(train,valid,test) = sequence_generator(original_data,look_back)
test_ground_truth = {i:test[i][1] for i in range(len(test))}

print("Train: {}, Valid: {}, Test: {}".format(len(train),len(valid),len(test)))

if not os.path.exists(output_path):
    os.makedirs(output_path)

print("Max Epochs:{}, Number of Ensenbles:{}".format(epochs, num_ensemble))
print("No data perturbation")

ensemble = Ensemble_LSTM(input_size=128, output_size=len(unique_items)+1, hidden_dim=64, n_layers=1, \
    num_ensemble=num_ensemble, device=device, initial = mlp_initial, bias = mlp_bias).to(device)
ensemble_eval, baseline_eval = ensemble.train(train=train, test= test, valid = valid, epochs = epochs)

np.save(output_path + "/acc_MLP_ensemble_" + str(num_ensemble) + "_set_init_"+ str(mlp_initial) +".npy", ensemble_eval)
np.save(output_path + "/acc_baseline_" + str(num_ensemble) + "_set_init_"+ str(mlp_initial) +".npy", baseline_eval)

print("Finished")

# In[5] 