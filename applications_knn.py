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

from helper import *
from helper_knn import *
from utility_func import *
from prepare_data import *
from if_utils import *
import config


# python applications_knn.py --task mislabel_detect --dataset creditcard --value_type KNN-SV-JW --n_data 2000 --n_val 2000 --flip_ratio 0.1 --K 5

# python applications_knn.py --task mislabel_detect --dataset creditcard --value_type WTNN-SV --n_data 200 --n_val 10 --flip_ratio 0.1 --tau -0.5
# python applications_knn.py --task mislabel_detect --dataset creditcard --value_type WKNN-SV --n_data 200 --n_val 10 --flip_ratio 0.1 --K 5


import argparse

parser = argparse.ArgumentParser('')

parser.add_argument('--dataset', type=str)
parser.add_argument('--value_type', type=str)
parser.add_argument('--n_data', type=int, default=500)
parser.add_argument('--n_val', type=int, default=2000)
parser.add_argument('--flip_ratio', type=float, default=0)
parser.add_argument('--task', type=str)
parser.add_argument('--dis_metric', type=str, default='cosine')
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--tau', type=float, default=0)

parser.add_argument('--sigma', type=float, default=0)
parser.add_argument('--q', type=float, default=1)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--eps', type=float, default=-1)
parser.add_argument('--q_test', type=float, default=1)
parser.add_argument('--val_corrupt', type=float, default=0)
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--normalizerow', action='store_true')

parser.add_argument('--kernel', type=str, default='plain') # uniform

# No use for this project
parser.add_argument('--model_type', type=str, default='')
parser.add_argument('--n_repeat', type=int, default=5)
parser.add_argument('--n_sample', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--card', type=int, default=0)
parser.add_argument('--last_epoch', action='store_false')
parser.add_argument('--random_state', type=int, default=1)

parser.add_argument('--noisydata', action='store_true')


args = parser.parse_args()

dataset = args.dataset
value_type = args.value_type
model_type = args.model_type
n_data = args.n_data
n_val = args.n_val
n_repeat = args.n_repeat
n_sample = args.n_sample
random_state = args.random_state
flip_ratio = float(args.flip_ratio) * 1.0
batch_size = args.batch_size
lr = args.lr
a, b = args.alpha, args.beta
task = args.task
card = args.card
K, tau = args.K, args.tau
sigma, q = args.sigma, args.q
delta = args.delta
eps = args.eps
q_test = args.q_test
dis_metric = args.dis_metric
big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'

verbose = 0
if args.debug:
  verbose = 1

batch_size = 32

if task != 'mislabel_detect':
  u_func = get_ufunc(dataset, model_type, batch_size, lr, verbose)


if task=='mislabel_detect':
  x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio, noisy_data=False, normalize=args.normalize)
elif task=='noisy_detect':
  x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio, noisy_data=True)


print(y_train)


if args.normalizerow and dataset in OpenML_dataset:
  x_train, y_train, x_val, y_val = get_processed_data_clip(dataset, n_data, n_val, flip_ratio)


if args.val_corrupt > 0:
  n_corrupt = int(n_val * args.val_corrupt)
  x_val[:n_corrupt] += np.random.normal(loc=10.0, scale=0.0, size=(n_corrupt, x_val.shape[1]))


# if value_type[:3] == 'TNN' and args.tau==0:
#   threshold = get_tuned_tau(x_train, y_train, x_val, y_val, dis_metric = dis_metric)
#   print('Tuned Tau:', threshold)
#   tau = threshold

# if value_type[:3] == 'KNN' and args.K==0:
#   threshold = get_tuned_K(x_train, y_train, x_val, y_val, dis_metric=dis_metric)
#   print('Tuned K:', threshold)
#   K = threshold



# for gamma in [0, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
#   print('gamma={}, acc={}'.format(
#     gamma, get_wtnn_acc(x_train, y_train, x_val, y_val, tau=-0.5, dis_metric='cosine', kernel='rbf', gamma=gamma)))



print('acc_TNN={}'.format(get_tnn_acc(x_train, y_train, x_val, y_val, tau=-0.5, dis_metric='cosine')))

knn_val_collection = ['KNN-SV', 'KNN-SV-RJ', 'KNN-SV-JW', 'TNN-BZ', 'TNN-BZ-private', 'TNN-SV', 
                      'TNN-SV-private', 'KNN-SV-RJ-private', 'KNN-SV-RJ-private-withsub', 'KNN-BZ-private-fixeps', 
                      'TNN-SV-private-JDP', 'WTNN-SV', 'fastWTNN-SV', 'WKNN-SV', 'fastWKNN-SV', 'approxfastWKNN-SV', 
                      'fastWKNN-SV-old']


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

sigma_use = -1

for i in range(n_repeat):

  if(random_state != -1): 
      np.random.seed(random_state+i)
      random.seed(random_state+i)

  v_args = copy.deepcopy(value_args)

  if value_type in ['Shapley_Perm', 'Banzhaf_GT', 'BetaShapley']:

    if dataset in big_dataset or dataset in OpenML_dataset :
      v_args['y_feature'] = value_args['y_feature']
    else:
      v_args['y_feature'] = np.clip( value_args['y_feature'] + np.random.normal(scale=args.noise, size=n_sample) , a_min=0, a_max=1)

  elif value_type == 'LOO':

    if dataset in big_dataset or dataset in OpenML_dataset :
      v_args['y_feature'] = value_args['y_feature']
      v_args['u_total'] = value_args['u_total']
    else:
      v_args['y_feature'] = np.clip( value_args['y_feature']+np.random.normal(scale=args.noise, size=len(value_args['y_feature'])), a_min=0, a_max=1)
      v_args['u_total'] = np.clip( value_args['u_total']+np.random.normal(scale=args.noise), a_min=0, a_max=1)

  elif value_type == 'FixedCard_MC':

    def func(x):
      if type(x[i]) is list:
        if args.last_epoch:
          return x[i][-1]
        else:
          return get_converge(x[i])
      else:
        return x[i]

    v_args['func'] = func

  elif value_type in ['FixedCard_MSR', 'FixedCard_MSRPerm']:
    
    def func(x):
      if type(x[i]) is list:
        if args.last_epoch:
          return x[i][-1]
        else:
          return get_converge(x[i])
      else:
        return x[i]

    v_args['func'] = func

  elif value_type in ['FZ20']:

    if args.dataset in OpenML_dataset:
      v_args['y_feature'] = value_args['y_feature'][:, i]
    else:
      v_args['y_feature'] = value_args['y_feature'][:, i, -1]


  start = time.time()

  if value_type == 'inf':
    sv = compute_influence_score(dataset, model_type, x_train, y_train, x_val, y_val, batch_size, lr, verbose=0)
  elif value_type == 'KNN-SV-RJ':
    sv = knn_shapley_RJ(x_train, y_train, x_val, y_val, K=K, dis_metric=dis_metric )
  elif value_type == 'KNN-SV-RJ-private':
    sv, eps, sigma_use = private_knn_shapley_RJ(x_train, y_train, x_val, y_val, K=K, sigma=sigma_use, delta=delta, q_test=q_test, dis_metric=dis_metric, eps=args.eps)
  elif value_type == 'KNN-SV-RJ-private-withsub':
    sv, eps, sigma_use = private_knn_shapley_RJ_withsub(x_train, y_train, x_val, y_val, K=K, sigma=sigma_use, q=q, delta=delta, q_test=q_test, dis_metric=dis_metric, eps=args.eps)
  elif value_type == 'KNN-SV-JW':
    sv = knn_shapley_JW(x_train, y_train, x_val, y_val, K=K, dis_metric = dis_metric)
  elif value_type == 'TNN-BZ':
    sv = knn_banzhaf(x_train, y_train, x_val, y_val, tau=tau, K0=K, dis_metric=dis_metric)
  elif value_type == 'TNN-SV':
    sv = tnn_shapley(x_train, y_train, x_val, y_val, tau=tau, K0=K, dis_metric=dis_metric)
  elif value_type == 'WTNN-SV':
    sv = weighted_tknn_shapley(x_train, y_train, x_val, y_val, tau=tau, dis_metric=dis_metric, kernel=args.kernel, debug=args.debug)
  elif value_type == 'fastWTNN-SV':
    sv = fastweighted_tknn_shapley(x_train, y_train, x_val, y_val, tau=tau, dis_metric=dis_metric, kernel=args.kernel, debug=args.debug)
  elif value_type == 'WKNN-SV':
    sv = weighted_knn_shapley(x_train, y_train, x_val, y_val, K=K, dis_metric=dis_metric, kernel=args.kernel, debug=args.debug)
  elif value_type == 'fastWKNN-SV':
    sv = fastweighted_knn_shapley(x_train, y_train, x_val, y_val, eps=args.eps, K=K, dis_metric=dis_metric, kernel=args.kernel, debug=args.debug)
  elif value_type == 'fastWKNN-SV-old':
    sv = fastweighted_knn_shapley_old(x_train, y_train, x_val, y_val, K=K, dis_metric=dis_metric, kernel=args.kernel, debug=args.debug)
  elif value_type == 'approxfastWKNN-SV':
    sv = approxfastweighted_knn_shapley(x_train, y_train, x_val, y_val, K=K, eps=eps, dis_metric=dis_metric, kernel=args.kernel, debug=args.debug)
  elif value_type == 'TNN-BZ-private':
    sv, eps, sigma_use = private_knn_banzhaf(x_train, y_train, x_val, y_val, tau=tau, K0=K, sigma=sigma_use, q=q, delta=delta, q_test=q_test, dis_metric = dis_metric)
  elif value_type == 'TNN-SV-private':
    sv, eps, sigma_use = private_tnn_shapley(x_train, y_train, x_val, y_val, tau=tau, K0=K, sigma=sigma_use, q=q, delta=delta, q_test=q_test, debug=args.debug, dis_metric = dis_metric)
  elif value_type == 'TNN-SV-private-JDP':
    sv, eps, sigma_use = private_tnn_shapley_JDP(x_train, y_train, x_val, y_val, tau=tau, K0=K, sigma=sigma_use, q=q, delta=delta, q_test=q_test, dis_metric=dis_metric, eps=args.eps)
  else:
    sv = compute_value(value_type, v_args)

  if value_type in ['WTNN-SV', 'fastWTNN-SV']:
    print('Data Value Computed; Value Name: {} ({}); Runtime: {} s'.format( value_type, args.kernel, np.round(time.time()-start, 3) ))
  else:
    print('Data Value Computed; Value Name: {}; Runtime: {} s'.format( value_type, np.round(time.time()-start, 3) ))


  if task=='weighted_acc':

    sv = normalize(sv) if value_type!='Uniform' else sv

    if dataset in big_dataset or dataset in OpenML_dataset:
      acc_lst = []
      for j in range(5):
        acc_lst.append( u_func(x_train, y_train, x_val, y_val, sv) )
      acc = np.mean(acc_lst)
    else:
      acc = u_func(x_train, y_train, x_val, y_val, sv)
    print('round {}, acc={}'.format(i, acc))
    data_lst.append( acc )
    

  elif task in ['mislabel_detect', 'noisy_detect']:
    acc1, acc2, auc = kmeans_f1score(sv, cluster=False), kmeans_f1score(sv, cluster=True), kmeans_aucroc(sv)
    data_lst.append( [acc1, acc2, auc] )

    
  elif task=='data_removal':

    rank = np.argsort(sv)
    acc_lst = []
    
    for k in np.linspace(0, int(args.n_data/2), num=11).astype(int):

      temp_lst = []
      for j in range(5):
        temp_lst.append( u_func(x_train[rank[k:]], y_train[rank[k:]], x_val, y_val) )
      acc = np.mean(temp_lst)
      print(acc)
      acc_lst.append(acc)
      
    print(acc_lst)
      
    data_lst.append(acc_lst)


  elif task=='data_add':

    rank = np.argsort(sv)[::-1]
    acc_lst = []

    # pdb.set_trace()
    
    for k in np.linspace(0, int(args.n_data/2), num=11).astype(int)[1:]:

      temp_lst = []
      for j in range(5):
        temp_lst.append( u_func(x_train[rank[:k]], y_train[rank[:k]], x_val, y_val) )
      acc = np.mean(temp_lst)
      acc_lst.append(acc)
      
    print(acc_lst)
      
    data_lst.append(acc_lst)

  elif task=='collect_sv':
    sv_collect.append(sv)



if task in ['mislabel_detect', 'noisy_detect']:

  print('Task: {}'.format(task))
  
  data_lst = np.array(data_lst)

  f1_rank, f1_rank_std = np.round( np.mean(data_lst[:, 0]), 3), np.round( np.std(data_lst[:, 0]), 3)
  f1_cluster, f1_cluster_std = np.round( np.mean(data_lst[:, 1]), 3), np.round( np.std(data_lst[:, 1]), 3)
  auc, std_auc = np.round( np.mean(data_lst[:, 2]), 3), np.round( np.std(data_lst[:, 2]), 3)

  if value_type == 'BetaShapley':
    print('*** {}_{}_{} {} ({}) {} ({}) ***'.format(value_type, a, b, f1_rank, f1_rank_std, f1_cluster, f1_cluster_std ))
  elif value_type in ['FixedCard_MC', 'FixedCard_MSR', 'FixedCard_MSRPerm']:
    print('*** {} card={} {} ({}) {} ({}) ***'.format(value_type, args.card, f1_rank, f1_rank_std, f1_cluster, f1_cluster_std ))
  else:
    print('*** {} F1-Rank: {} ({}), F1-Cluster: {} ({}), AUROC: {} ({}), eps={}, delta={}, K={}, tau={} ***'.format(
      value_type, f1_rank, f1_rank_std, f1_cluster, f1_cluster_std, auc, std_auc, eps, delta, K, tau))
    
elif task == 'data_removal':
  
  file_name = 'DATAREMOVAL_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.result'.format(args.dataset, args.value_type, args.model_type, args.n_data, args.n_val, 
                                                                           args.n_repeat, args.n_sample, args.flip_ratio, args.alpha, args.beta, args.card )
  
  pickle.dump( data_lst, open(save_dir + file_name, 'wb') )

elif task == 'data_add':
  
  file_name = 'DATAADD_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.result'.format(args.dataset, args.value_type, args.model_type, args.n_data, args.n_val, 
                                                                           args.n_repeat, args.n_sample, args.flip_ratio, args.alpha, args.beta, args.card )
  
  pickle.dump( data_lst, open(save_dir + file_name, 'wb') )

elif task == 'collect_sv':

  file_name = 'SV_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.result'.format(args.dataset, args.value_type, args.model_type, args.n_data, args.n_val, 
                                                                           args.n_repeat, args.n_sample, args.flip_ratio, args.alpha, args.beta, args.card )
  
  pickle.dump( sv_collect, open(save_dir + file_name, 'wb') )

