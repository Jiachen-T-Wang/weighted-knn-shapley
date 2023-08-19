import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

# general
import pandas as pd 
import numpy as np 
import copy
import pickle
import sys
import time
import os
import random
import pdb

from scipy.stats import norm

from helper import *
from helper_knn import *
from utility_func import *
from prepare_data import *
from if_utils import *
import config

import argparse

# python datavalue-privleak.py --dataset cpu --value_type KNN-SV-JW --n_data 200 --n_val 200 --flip_ratio 0 --random_state 1 --K 5

# python datavalue-mia.py --dataset cpu --value_type KNN-SV-JW --n_query 200 --n_dist 400 --n_val 200 --n_sample 32 --n_atk 20 --flip_ratio 0 --random_state 1 --K 5


parser = argparse.ArgumentParser('')

parser.add_argument('--dataset', type=str)
parser.add_argument('--value_type', type=str)
parser.add_argument('--n_query', type=int, default=500)
parser.add_argument('--n_val', type=int, default=2000)
parser.add_argument('--n_dist', type=int, default=2000)

parser.add_argument('--n_sample', type=int)
parser.add_argument('--n_atk', type=int)


parser.add_argument('--n_repeat', type=int, default=5)
parser.add_argument('--random_state', type=int)
parser.add_argument('--flip_ratio', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--card', type=int, default=0)
parser.add_argument('--last_epoch', action='store_false')

parser.add_argument('--K', type=int, default=5)
parser.add_argument('--tau', type=float, default=0)
parser.add_argument('--sigma', type=float, default=0)
parser.add_argument('--q', type=float, default=1)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--eps', type=float, default=0)
parser.add_argument('--q_test', type=float, default=1)


args = parser.parse_args()

dataset = args.dataset
value_type = args.value_type

n_data = args.n_query + args.n_dist

n_val = args.n_val
n_repeat = args.n_repeat
n_sample = args.n_sample
random_state = args.random_state
flip_ratio = float(args.flip_ratio) * 1.0
batch_size = args.batch_size
lr = args.lr
a, b = args.alpha, args.beta
card = args.card

K, tau = args.K, args.tau
sigma, q = args.sigma, args.q
delta = args.delta
eps = np.infty
q_test = args.q_test

n_atk = args.n_atk

big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'privleak/'

verbose = 0
if args.debug:
  verbose = 1


x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio)

x_query, y_query = x_train[:args.n_query], y_train[:args.n_query]
x_query_in, y_query_in = x_query[:int(0.5*args.n_query)], y_query[:int(0.5*args.n_query)]
x_query_out, y_query_out = x_query[int(0.5*args.n_query):], y_query[int(0.5*args.n_query):]
x_dist, y_dist = x_train[args.n_query:], y_train[args.n_query:]


print(x_query_in.shape)


if(random_state != -1): 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


knn_val_collection = ['KNN-SV', 'KNN-BZ', 'KNN-SV-RJ', 'KNN-SV-JW', 'KNN-BZ-private', 'KNN-SV-RJ-private', 'KNN-BZ-private-fixeps']


if value_type != 'Uniform' and value_type != 'inf' and value_type not in knn_val_collection:
  value_args = load_value_args(value_type, args)
  value_args['n_data'] = n_data
  if dataset in big_dataset:
    value_args['sv_baseline'] = 0.1
  else:
    value_args['sv_baseline'] = 0.5
else:
  value_args = {}
  value_args['n_data'] = n_data


data_lst = []

sv_collect = []


if value_type == 'KNN-SV-RJ':
   knn_shapley = knn_shapley_RJ
elif value_type == 'KNN-SV-JW':
   knn_shapley = knn_shapley_JW
else:
   exit(1)



def get_in_out_value(x_sample, y_sample, x, y):
    x_sample_aug = np.insert(x_sample, 0, x, axis=0)
    y_sample_aug = np.insert(y_sample, 0, y)
    sv_out = knn_shapley(x_sample_aug, y_sample_aug, x_val, y_val, K=K)
    sv_out = sv_out[0]

    print(sv_out)
    # print(x_sample_aug.shape)

    x_sample_aug = np.insert(x_sample_aug, 0, x, axis=0)
    y_sample_aug = np.insert(y_sample_aug, 0, y) 
    sv_in = knn_shapley(x_sample_aug, y_sample_aug, x_val, y_val, K=K)
    # print(sv_in[0], sv_in[1])
    sv_in = sv_in[0]

    # print(x_sample_aug.shape)

    print(sv_in, sv_out)

    return np.abs(sv_in), np.abs(sv_out)


# Collect IN and OUT Data Values
sv_IN = np.zeros((2*n_atk, n_sample))
sv_OUT = np.zeros((2*n_atk, n_sample))


print('Collect IN and OUT Data Values')
for i in tqdm(range( n_sample )):

  idx = uniformly_subset_sample(np.arange(len(y_dist)))
  x_sample, y_sample = x_dist[idx], y_dist[idx]

  # Query n_atk data points IN
  for j in range(n_atk):
    x, y = x_query_in[j], y_query_in[j]
    sv_in, sv_out = get_in_out_value( x_sample, y_sample, x, y)
    sv_IN[j, i] = sv_in
    sv_OUT[j, i] = sv_out

  # Query n_atk data points OUT
  for j in range(n_atk):
    x, y = x_query_out[j], y_query_out[j]
    sv_in, sv_out = get_in_out_value( x_sample, y_sample, x, y)
    sv_IN[j+n_atk, i] = sv_in
    sv_OUT[j+n_atk, i] = sv_out


print('Construct Groudtruth')
ground_truth = np.ones(2*n_atk)
ground_truth[n_atk:] = 0

print('Attack')
likelihood_ratio = np.zeros(2*n_atk)
for j in tqdm(range(n_atk)):
    
    x, y = x_query_in[j], y_query_in[j]
    x_train_aug = np.insert(x_query_in, 0, x, axis=0)
    y_train_aug = np.insert(y_query_in, 0, y)
    sv_obs = knn_shapley(x_train_aug, y_train_aug, x_val, y_val, K=K)
    sv_obs = np.abs(sv_obs[0])
    lld_in = norm.pdf( sv_obs, loc=np.mean(sv_IN[j, :]), scale=np.std(sv_IN[j, :]) )
    lld_out = norm.pdf( sv_obs, loc=np.mean(sv_OUT[j, :]), scale=np.std(sv_OUT[j, :]) )
    lld = lld_in / lld_out
    likelihood_ratio[j] = lld

    x, y = x_query_out[j], y_query_out[j]
    x_train_aug = np.insert(x_query_in, 0, x, axis=0)
    y_train_aug = np.insert(y_query_in, 0, y)
    sv_obs = knn_shapley(x_train_aug, y_train_aug, x_val, y_val, K=K)
    sv_obs = np.abs(sv_obs[0])
    lld_in = norm.pdf( sv_obs, loc=np.mean(sv_IN[j+n_atk, :]), scale=np.std(sv_IN[j+n_atk, :]) )
    lld_out = norm.pdf( sv_obs, loc=np.mean(sv_OUT[j+n_atk, :]), scale=np.std(sv_OUT[j+n_atk, :]) )
    lld = lld_in / lld_out
    likelihood_ratio[j+n_atk] = lld

# print(likelihood_ratio)
print('K: {}, Attack AUROC: {}'.format(K, roc_auc_score(ground_truth, likelihood_ratio)))
