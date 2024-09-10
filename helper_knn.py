import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import time

# general
import pandas as pd 
import numpy as np 
import copy
import pickle
import sys
import time
import os
from os.path import exists
import warnings

from tqdm import tqdm

import scipy
from scipy.special import beta, comb
from scipy.spatial import distance
from random import randint

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


import config


big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'


def rank_neighbor(x_test, x_train, dis_metric='cosine'):
  if dis_metric == 'cosine':
    distance = -np.dot(x_train, x_test) / np.linalg.norm(x_train, axis=1)
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train])
  return np.argsort(distance)






# x_test, y_test are single data point
def knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric='cosine'):

  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few, dis_metric = dis_metric)
  sv[int(rank[-1])] += int(y_test==y_train_few[int(rank[-1])]) / N

  for j in range(2, N+1):
    i = N+1-j
    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + ( (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / K ) * min(K, i) / i

  return sv


# Original KNN-Shapley proposed in http://www.vldb.org/pvldb/vol12/p1610-jia.pdf
def knn_shapley_RJ(x_train_few, y_train_few, x_val_few, y_val_few, K, dis_metric='cosine', collect_sv=False):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  sv_lst = []

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv0 = knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric=dis_metric)
    sv += sv0

    sv_lst.append(sv0)

  if collect_sv:
    return sv, sv_lst
  else:
    return sv


# x_test, y_test are single data point
def knn_shapley_JW_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric='cosine'):
  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few, dis_metric=dis_metric).astype(int)
  C = max(y_train_few)+1

  c_A = np.sum( y_test==y_train_few[rank[:N-1]] )

  const = np.sum([ 1/j for j in range(1, min(K, N)+1) ])

  sv[rank[-1]] = (int(y_test==y_train_few[rank[-1]]) - c_A/(N-1)) / N * ( np.sum([ 1/(j+1) for j in range(1, min(K, N)) ]) ) + (int(y_test==y_train_few[rank[-1]]) - 1/C) / N

  for j in range(2, N+1):
    i = N+1-j
    coef = (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / (N-1)

    sum_K3 = K

    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + coef * ( const + int( N >= K ) / K * ( min(i, K)*(N-1)/i - sum_K3 ) )

  return sv


# Soft-label KNN-Shapley proposed in https://arxiv.org/abs/2304.04258 
def knn_shapley_JW(x_train_few, y_train_few, x_val_few, y_val_few, K, dis_metric='cosine'):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += knn_shapley_JW_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric = dis_metric)

  return sv


def get_knn_acc(x_train, y_train, x_val, y_val, K, dis_metric='l2'):
  n_val = len(y_val)
  C = max(y_train)+1

  acc = 0

  for i in range(n_val):
    x_test, y_test = x_val[i], y_val[i]
    if dis_metric == 'cosine':
      distance = -np.dot(x_train, x_test)
    else:
      distance = np.array([np.linalg.norm(x - x_test) for x in x_train])
    rank = np.argsort(distance)
    acc_single = 0
    for j in range(K):
      acc_single += int(y_test==y_train[ rank[j] ])
    acc += (acc_single/K)

  return acc / n_val


import itertools

def check_runtime_baseline(x_train, y_train, x_val, y_val, K, dis_metric='l2'):
    # Generate indices of the original dataset

    start = time.time()

    indices = list(range(len(x_train)))
    
    # Loop through all possible subset sizes
    for size in range(1, K + 1):
        # Generate combinations of indices
        for subset_indices in itertools.combinations(indices, size):
            # Extract the corresponding data points
            subset_x = np.array( [x_train[i] for i in subset_indices] )
            subset_y = np.array( [y_train[i] for i in subset_indices] )
            acc = get_knn_acc(x_train, y_train, x_val, y_val, K, dis_metric)
    print('Runtime: {}'.format(time.time() - start))



# Return the utility of WKNN on a single point
# mode: softlabel or hardlabel
# x_val, y_val: a single test point
def weighted_knn_accuracy(x_train, y_train, x_val, y_val, K, dis_metric, kernel, mode):
    
    # Assume that the utility of empty set is 0.
    if len(y_train) == 0:
      return 0
    
    distances = compute_dist(x_train, x_val, dis_metric)
    weights = compute_weights(distances, kernel=kernel)
    sorted_indices = np.argsort(distances)

    if len(y_train) < K:
      k_nearest_indices = sorted_indices
    else:
      k_nearest_indices = sorted_indices[:K]

    k_nearest_labels = y_train[k_nearest_indices]
    weights = weights[k_nearest_indices]

    # For soft-label: Normalize KNN's weights
    if mode == 'softlabel':
      weights /= np.sum(weights)
      sum_weights = sum(weight for weight, label in zip(weights, k_nearest_labels) if label == y_val)
      utility = sum_weights

    elif mode == 'hardlabel':
      unique_labels = np.unique(k_nearest_labels)
      weight_sum = [np.sum(weights[k_nearest_labels == label]) for label in unique_labels]
      predicted_label = unique_labels[np.argmax(weight_sum)]
      utility = predicted_label == y_val

    return utility



# Implement the baseline algorithm from Jia et al. (2019)
def WKNNSV_RJ_singlepoint(x_train, y_train, x_val, y_val, K, dis_metric, kernel, mode):

  N = len(y_train)
  sv = np.zeros(N)
  
  indices = list(range(len(x_train)))
  distances = compute_dist(x_train, x_val, dis_metric)
  # weights = compute_weights(distances, kernel=kernel)
  rank = np.argsort(distances)

  ind_N = rank[-1]

  indices_looN = [index for index in indices if index != ind_N]

  # compute phi_N
  for size in range(0, K):
    # Generate combinations of indices
    for subset_indices in itertools.combinations(indices_looN, size):

      subset_x_without = np.array( [x_train[i] for i in subset_indices] )
      subset_y_without = np.array( [y_train[i] for i in subset_indices] )
      acc_without = weighted_knn_accuracy(subset_x_without, subset_y_without, x_val, y_val, K, dis_metric, kernel, mode)

      subset_x_with = np.array( [x_train[i] for i in subset_indices] + [x_train[ind_N]] )
      subset_y_with = np.array( [y_train[i] for i in subset_indices] + [y_train[ind_N]] )
      acc_with = weighted_knn_accuracy(subset_x_with, subset_y_with, x_val, y_val, K, dis_metric, kernel, mode)

      sv[ind_N] += (acc_with - acc_without) / (N*comb(N-1, size))

  # recursively compute phi_j
  for j in range(2, N+1):

    ind_Jp1, ind_J = rank[-(j-1)], rank[-j]
    indices_leave_two_out = [index for index in indices if index not in [ind_Jp1, ind_J]]

    cumu = 0

    for size in range(0, K-1):
      cumu_size = 0
      for subset_indices in itertools.combinations(indices_leave_two_out, size):
        subset_x_withJp1 = np.array( [x_train[i] for i in subset_indices] + [x_train[ind_Jp1]] )
        subset_y_withJp1 = np.array( [y_train[i] for i in subset_indices] + [y_train[ind_Jp1]] )
        acc_withJp1 = weighted_knn_accuracy(subset_x_withJp1, subset_y_withJp1, x_val, y_val, K, dis_metric, kernel, mode)
        subset_x_withJ = np.array( [x_train[i] for i in subset_indices] + [x_train[ind_J]] )
        subset_y_withJ = np.array( [y_train[i] for i in subset_indices] + [y_train[ind_J]] )
        acc_withJ = weighted_knn_accuracy(subset_x_withJ, subset_y_withJ, x_val, y_val, K, dis_metric, kernel, mode)
        cumu_size += (acc_withJ - acc_withJp1)
      cumu_size /= comb(N-2, size)
      cumu += cumu_size

    for size in range(K-1, N-1):
      cumu_size = 0
      for subset_indices in itertools.combinations(indices_leave_two_out, K-1):
        subset_x_withJp1 = np.array( [x_train[i] for i in subset_indices] + [x_train[ind_Jp1]] )
        subset_y_withJp1 = np.array( [y_train[i] for i in subset_indices] + [y_train[ind_Jp1]] )
        acc_withJp1 = weighted_knn_accuracy(subset_x_withJp1, subset_y_withJp1, x_val, y_val, K, dis_metric, kernel, mode)
        subset_x_withJ = np.array( [x_train[i] for i in subset_indices] + [x_train[ind_J]] )
        subset_y_withJ = np.array( [y_train[i] for i in subset_indices] + [y_train[ind_J]] )
        acc_withJ = weighted_knn_accuracy(subset_x_withJ, subset_y_withJ, x_val, y_val, K, dis_metric, kernel, mode)

        i = N+1-j
        r = i+1

        if K > 1:
          subset_indices = list(subset_indices)
          a = np.argmax( distances[subset_indices] )
          ind_max = subset_indices[a]
          r = max(r, np.where(rank == ind_max)[0] + 1)
        cumu_size += (acc_withJ - acc_withJp1) * comb(N-r, size-(K-1))

      cumu_size /= comb(N-2, size)
      cumu += cumu_size

    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + cumu / (N-1)

  print('Sanity check: sum of SV = {}, U(N)-U(empty)={}'.format(
      np.sum(sv), weighted_knn_accuracy(x_train, y_train, x_val, y_val, K, dis_metric, kernel, mode) )
  )

  return sv


# Soft-label KNN-Shapley proposed in https://arxiv.org/abs/2304.04258 
def WKNNSV_RJ(x_train_few, y_train_few, x_val_few, y_val_few, K, dis_metric, kernel, mode):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += WKNNSV_RJ_singlepoint(x_train_few, y_train_few, x_test, y_test, K, dis_metric, kernel, mode)

  return sv


# Implement Dummy Data Point Idea
def WKNNSV_MC_RJ_perm(x_train, y_train, x_val, y_val, K, dis_metric, kernel, mode, n_sample):

  X_feature_test = []
  y_feature_test = []

  n_data = len(y_train)
  n_perm = int(n_sample / n_data)
  
  for k in range(n_perm):

    print('Permutation {} / {}'.format(k, n_perm))
    perm = np.random.permutation(range(n_data))

    for i in range(0, n_data+1):
      subset_index = perm[:i]
      X_feature_test.append(subset_index)
      y_feature_test.append(weighted_knn_accuracy(x_train[subset_index], y_train[subset_index], x_val, y_val, 
                                                  K, dis_metric, kernel, mode))

  return X_feature_test, y_feature_test


def shapley_permsampling_from_data(X_feature, y_feature, n_data):

  n_sample = len(y_feature)
  n_perm = int( n_sample // n_data ) - 1

  if n_sample%n_data > 0: 
    print('WARNING: n_sample cannot be divided by n_data')

  sv_vector = np.zeros(n_data)

  for i in range(n_perm):
    for j in range(1, n_data+1):
      target_ind = X_feature[i*(n_data+1)+j][-1]
      without_score = y_feature[i*(n_data+1)+j-1]
      with_score = y_feature[i*(n_data+1)+j]
      
      sv_vector[target_ind] += (with_score-without_score)
  
  return sv_vector / n_perm


# Soft-label KNN-Shapley proposed in https://arxiv.org/abs/2304.04258 
def WKNNSV_MC_RJ(x_train_few, y_train_few, x_val_few, y_val_few, K, dis_metric, kernel, mode, n_sample):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    X_feature, y_feature = WKNNSV_MC_RJ_perm(x_train_few, y_train_few, x_test, y_test, 
                                             K, dis_metric, kernel, mode, n_sample)
    sv_i = shapley_permsampling_from_data(X_feature, y_feature, N)
    sv += sv_i

  return sv





def get_tuned_K(x_train, y_train, x_val, y_val, dis_metric='cosine'):

  acc_max = 0
  best_K = 0

  for K in range(1, 8):
    acc = get_knn_acc(x_train, y_train, x_val, y_val, K, dis_metric = dis_metric)
    print('K={}, acc={}'.format(K, acc))
    if acc > acc_max:
      acc_max = acc
      best_K = K

  return best_K


def get_tnn_acc(x_train, y_train, x_val, y_val, tau, dis_metric='cosine'):
  n_val = len(y_val)
  C = max(y_train)+1
  acc = 0
  for i in range(n_val):
    x_test, y_test = x_val[i], y_val[i]
    #ix_test = x_test.reshape((-1,1))
    if dis_metric == 'cosine':
      distance = - np.dot(x_train, x_test) / np.linalg.norm(x_train, axis=1)
    else:
      distance = np.array([np.linalg.norm(x - x_test) for x in x_train])
    Itau = (distance<tau).nonzero()[0]
    acc_single = 0
    #print(f'tune tau size of Tau is {len(Itau)}')
    if len(Itau) > 0:
      for j in Itau:
        acc_single += int(y_test==y_train[j])
      acc_single = acc_single / len(Itau)
    else:
      acc_single = 1/C
    acc += acc_single
  return acc / n_val


def get_tuned_tau(x_train, y_train, x_val, y_val, dis_metric='cosine'):

  print('dis_metric', dis_metric)
  acc_max = 0
  best_tau = 0
  # because we use the negative cosine value as the distance metric
  tau_list =[-0.04*x for x in range(25)]+[0.04*x for x in range(10)]
  for tau in tau_list:
    acc = get_tnn_acc(x_train, y_train, x_val, y_val, tau, dis_metric=dis_metric)
    print('tau={}, acc={}'.format(tau, acc))
    if acc > acc_max:
      acc_max = acc
      best_tau = tau

  if best_tau == 1:
    for tau in (np.arange(1, 10) / 10):
      acc = get_tnn_acc(x_train, y_train, x_val, y_val, tau, dis_metric=dis_metric)
      print('tau={}, acc={}'.format(tau, acc))
      if acc > acc_max:
        acc_max = acc
        best_tau = tau

  return best_tau





"""
def get_tuned_tau(x_train, y_train, x_val, y_val):

  acc_max = 0
  best_tau = 0

  for tau in range(1, 11):
    acc = get_tnn_acc(x_train, y_train, x_val, y_val, tau)
    print('tau={}, acc={}'.format(tau, acc))
    if acc > acc_max:
      acc_max = acc
      best_tau = tau

  if best_tau == 1:
    for tau in (np.arange(1, 10) / 10):
      acc = get_tnn_acc(x_train, y_train, x_val, y_val, tau)
      print('tau={}, acc={}'.format(tau, acc))
      if acc > acc_max:
        acc_max = acc
        best_tau = tau

  return best_tau
"""


# x_test, y_test are single data point
def tnn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau=0, K0=10, dis_metric='cosine'):

  N = len(y_train_few)
  sv = np.zeros(N)

  C = max(y_train_few)+1
  if dis_metric == 'cosine':
    distance = -np.dot(x_train_few, x_test) / np.linalg.norm( x_train_few, axis=1 )
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])
  Itau = (distance < tau).nonzero()[0]

  Ct = len(Itau)
  Ca = np.sum( y_train_few[Itau] == y_test )

  reusable_sum = 0
  stable_ratio = 1
  for j in range(N):
    stable_ratio *= (N-j-Ct) / (N-j)
    reusable_sum += (1/(j+1)) * (1 - stable_ratio)
    # reusable_sum += (1/(j+1)) * (1 - comb(N-1-j, Ct) / comb(N, Ct))

  for i in Itau:
    sv[i] = ( int(y_test==y_train_few[i]) - 1/C ) / Ct
    if Ct >= 2:
      ca = Ca - int(y_test==y_train_few[i])
      sv[i] += ( int(y_test==y_train_few[i])/Ct - ca/(Ct*(Ct-1)) ) * ( reusable_sum - 1 )

  return sv

def tnn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, tau=0, K0=10, dis_metric='cosine'):
  
  N = len(y_train_few)
  sv = np.zeros(N)
  n_test = len(y_val_few)
  print('tau in tnn shapley', tau)
  for i in tqdm(range(n_test)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += tnn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau, K0, dis_metric=dis_metric)

  return sv



"""
def private_tnn_shapley_single_divq(x_train_few, y_train_few, x_test, y_test, tau=0, K0=10, sigma=0, q=1):

  N = len(y_train_few)
  sv = np.zeros(N)
  C = max(y_train_few)+1

  # Poisson Subsampling
  sub_ind_bool = (np.random.choice([0, 1], size=N, p=[1-q, q])).astype(bool)
  sub_ind = np.where(sub_ind_bool)[0]

  distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])
  rank_all = np.argsort(distance)

  if tau == 0:
    tau = x_train_few[rank_all[K0-1]]
  Itau_all = (distance <= tau).nonzero()[0]

  # Itau_subset: index in terms of subset
  distance_subset = distance[sub_ind]
  Itau_subset = (distance_subset <= tau).nonzero()[0]

  Ct = (len(Itau_subset) + np.random.normal(scale=sigma)) / q
  Ca = ( np.sum( y_train_few[sub_ind[Itau_subset]] == y_test ) + np.random.normal(scale=sigma) ) / q

  Ct, Ca = np.round(Ct), np.round(Ca)
  Ct, Ca = max(Ct, 0), max(Ca, 0)
  
  reusable_sum_i_in_sub = 0
  stable_ratio = 1
  for j in range( N ):
    stable_ratio *= (N-j-max(1, Ct)) / (N-j)
    reusable_sum_i_in_sub += (1/(j+1)) * (1 - stable_ratio)

  reusable_sum_i_notin_sub = 0
  stable_ratio = 1
  for j in range( N ):
    stable_ratio *= (N-j-(Ct+1)) / (N-j)
    reusable_sum_i_notin_sub += (1/(j+1)) * (1 - stable_ratio)


  for i in range(N):

    if i in Itau_all:

      if i in sub_ind:
        reusable_sum = reusable_sum_i_in_sub
        Ct_i = max(1, Ct)
        Ca_i = Ca
        if y_test==y_train_few[i]:
          Ca_i = max(1, Ca_i)
      else:
        reusable_sum = reusable_sum_i_notin_sub
        Ct_i = Ct + 1
        Ca_i = Ca + int(y_test==y_train_few[i])

      sv[i] = ( int(y_test==y_train_few[i]) - 1/C ) / Ct_i
      if Ct_i >= 2:
        ca = Ca_i - int(y_test==y_train_few[i])
        sv[i] += ( int(y_test==y_train_few[i])/Ct_i - ca/(Ct_i*(Ct_i-1)) ) * ( reusable_sum - 1 )

  return sv, (distance<=tau).astype(int)
"""


def private_tnn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau=0, K0=10, sigma=0, q=1, dis_metric='cosine'):

  N = len(y_train_few)
  sv = np.zeros(N)
  C = max(y_train_few)+1

  # Poisson Subsampling
  sub_ind_bool = (np.random.choice([0, 1], size=N, p=[1-q, q])).astype(bool)
  sub_ind = np.where(sub_ind_bool)[0]
  #print('which distance do we use', dis_metric)

  if dis_metric == 'cosine':
    distance = -np.dot(x_train_few, x_test) / np.linalg.norm( x_train_few, axis=1 )
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])

  Itau_all = (distance <= tau).nonzero()[0]

  # Itau_subset: index in terms of subset
  distance_subset = distance[sub_ind]
  Itau_subset = (distance_subset <= tau).nonzero()[0]

  Ct = len(Itau_subset) + np.random.normal(scale=sigma)
  Ca = np.sum( y_train_few[sub_ind[Itau_subset]] == y_test ) + np.random.normal(scale=sigma)

  Ct, Ca = np.round(Ct), np.round(Ca)
  Ct, Ca = max(Ct, 0), max(Ca, 0)

  # N_subset also needs to be privatized
  N_subset = len(sub_ind)
  N_subset = np.round( N_subset + np.random.normal(scale=sigma) )
  N_subset = int( max(N_subset, 0) )

  reusable_sum_i_in_sub = 0
  stable_ratio = 1
  for j in range(N_subset):
    stable_ratio *= (N_subset-j-max(1, Ct)) / (N_subset-j)
    reusable_sum_i_in_sub += (1/(j+1)) * (1 - stable_ratio)

  reusable_sum_i_notin_sub = 0
  stable_ratio = 1
  for j in range(N_subset+1):
    stable_ratio *= (N_subset+1-j-(Ct+1)) / (N_subset+1-j)
    reusable_sum_i_notin_sub += (1/(j+1)) * (1 - stable_ratio)

  count = 0

  for i in Itau_all:

    if i in Itau_all:

      if i in sub_ind:
        reusable_sum = reusable_sum_i_in_sub
        Ct_i = max(1, Ct)
        Ca_i = Ca
        if y_test==y_train_few[i]:
          Ca_i = max(1, Ca_i)
      else:
        reusable_sum = reusable_sum_i_notin_sub
        Ct_i = Ct + 1
        Ca_i = Ca + int(y_test==y_train_few[i])

      sv[i] = ( int(y_test==y_train_few[i]) - 1/C ) / Ct_i
      if Ct_i >= 2:
        ca = Ca_i - int(y_test==y_train_few[i])
        sv[i] += ( int(y_test==y_train_few[i])/Ct_i - ca/(Ct_i*(Ct_i-1)) ) * ( reusable_sum - 1 )
        count += 1

  return sv, (distance<tau).astype(int)


def private_tnn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, tau=0, K0=10, sigma=0, q=1, delta=1e-5, q_test=0.1, debug=False, dis_metric='cosine', rdp=False):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  n_test_sub = int(n_test*q_test)
  test_ind = np.random.choice(range(n_test), size=n_test_sub, replace=False)
  x_val_few, y_val_few = x_val_few[test_ind], y_val_few[test_ind]

  n_iter_lst = np.zeros(N)

  for i in range(n_test_sub):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv_individual, close_lst = private_tnn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau, K0, sigma*np.sqrt(3), q, dis_metric=dis_metric)
    sv += sv_individual
    n_iter_lst += close_lst


  # First run RDP and get a rough estimate of eps
  n_compose = np.round( np.mean(n_iter_lst) ).astype(int)
  mech = PrivateKNN_mech(q, sigma, n_compose)
  eps = mech.get_approxDP(delta=delta)

  # If eps estimate is too large or too small, use RDP
  if rdp or eps>30 or eps<0.01:
    print('use rdp')
    if debug:
      n_compose = np.round( np.mean(n_iter_lst) ).astype(int)
      mech = PrivateKNN_mech(q, sigma, n_compose)
      eps = mech.get_approxDP(delta=delta)
      return sv, eps
    else:
      eps_lst = np.zeros(N)
      for i, n_compose in tqdm(enumerate(n_iter_lst)):
        if n_compose == 0:
          eps_lst[i] = 0
        else:
          mech = PrivateKNN_mech(q, sigma, n_compose)
          eps = mech.get_approxDP(delta=delta)
          eps_lst[i] = eps
      return sv, (np.mean(eps_lst), np.max(eps_lst))

  else:

    prv = PoissonSubsampledGaussianMechanism(sampling_probability=q, noise_multiplier=sigma)
    acct = PRVAccountant(prvs=prv, max_self_compositions=n_test_sub, eps_error=1e-3, delta_error=1e-10)

    if debug:
      n_compose = np.round( np.mean(n_iter_lst) ).astype(int)
      low, est, upp = acct.compute_epsilon(delta=delta, num_self_compositions=[n_compose])
      return sv, upp
    else:
      eps_lst = np.zeros(N)
      for i, n_compose in enumerate(n_iter_lst):
        n_compose = n_compose.astype(int)
        if n_compose == 0:
          eps_lst[i] = 0
        else:
          low, est, upp = acct.compute_epsilon(delta=delta, num_self_compositions=[n_compose])
          eps_lst[i] = upp
      return sv, (np.mean(eps_lst), np.max(eps_lst))






def private_tnn_shapley_single_JDP(x_train_few, y_train_few, x_test, y_test, Nsubsethat, tau=0, K0=10, sigma=0, q=1, dis_metric='cosine'):

  N = len(y_train_few)
  sv = np.zeros(N)
  C = max(y_train_few)+1

  # Poisson Subsampling
  sub_ind_bool = (np.random.choice([0, 1], size=N, p=[1-q, q])).astype(bool)
  sub_ind = np.where(sub_ind_bool)[0]
  if dis_metric == 'cosine':
    distance = -np.dot(x_train_few, x_test) / np.linalg.norm( x_train_few, axis=1 )
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])

  Itau_all = (distance <= tau).nonzero()[0]

  # Itau_subset: index in terms of subset
  distance_subset = distance[sub_ind]
  Itau_subset = (distance_subset <= tau).nonzero()[0]

  Ct = len(Itau_subset) + np.random.normal(scale=sigma)
  Ca = np.sum( y_train_few[sub_ind[Itau_subset]] == y_test ) + np.random.normal(scale=sigma)

  Ct, Ca = np.round(Ct), np.round(Ca)
  Ct, Ca = max(Ct, 0), max(Ca, 0)

  # N_subset = len(sub_ind)
  N_subset = Nsubsethat

  reusable_sum_i_in_sub = 0
  stable_ratio = 1
  for j in range(N_subset):
    stable_ratio *= (N_subset-j-max(1, Ct)) / (N_subset-j)
    reusable_sum_i_in_sub += (1/(j+1)) * (1 - stable_ratio)

  reusable_sum_i_notin_sub = 0
  stable_ratio = 1
  for j in range(N_subset+1):
    stable_ratio *= (N_subset+1-j-(Ct+1)) / (N_subset+1-j)
    reusable_sum_i_notin_sub += (1/(j+1)) * (1 - stable_ratio)

  for i in Itau_all:

      if i in sub_ind:
        reusable_sum = reusable_sum_i_in_sub
        Ct_i = max(1, Ct)
        Ca_i = Ca
        if y_test==y_train_few[i]:
          Ca_i = max(1, Ca_i)
      else:
        reusable_sum = reusable_sum_i_notin_sub
        Ct_i = Ct + 1
        Ca_i = Ca + int(y_test==y_train_few[i])

      sv[i] = ( int(y_test==y_train_few[i]) - 1/C ) / Ct_i
      if Ct_i >= 2:
        ca = Ca_i - int(y_test==y_train_few[i])
        sv[i] += ( int(y_test==y_train_few[i])/Ct_i - ca/(Ct_i*(Ct_i-1)) ) * ( reusable_sum - 1 )

  return sv


def private_tnn_shapley_JDP(x_train_few, y_train_few, x_val_few, y_val_few, tau=0, K0=10, sigma=0, q=1, delta=1e-5, q_test=0.1, dis_metric='cosine', rdp=False, eps=-1):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  n_test_sub = int(n_test*q_test)
  test_ind = np.random.choice(range(n_test), size=n_test_sub, replace=False)
  x_val_few, y_val_few = x_val_few[test_ind], y_val_few[test_ind]

  n_compose = n_test_sub + 1

  if eps>0 and sigma<0:
    sigma = find_noise_multiplier(sampling_probability=q, num_steps=n_compose, target_epsilon=eps, target_delta=delta, eps_error=1e-2, mu_max=5000)
    print('sigma={}'.format(sigma))
  elif eps<0:
    # First run RDP and get a rough estimate of eps
    # n_compose+1 since we need to count for the noisy N_subset
    mech = PrivateKNN_mech(q, sigma, n_compose)
    eps = mech.get_approxDP(delta=delta)

    # If eps estimate is too large or too small, use RDP
    if rdp or eps>30 or eps<0.01:
      print('Use RDP')
    else:
      print('Use PRV')
      prv = PoissonSubsampledGaussianMechanism(sampling_probability=q, noise_multiplier=sigma)
      acct = PRVAccountant(prvs=prv, max_self_compositions=n_compose, eps_error=1e-3, delta_error=1e-10)
      low, est, upp = acct.compute_epsilon(delta=delta, num_self_compositions=[n_compose])
      eps = upp
  else:
    pass
  
  sub_ind_bool = (np.random.choice([0, 1], size=N, p=[1-q, q])).astype(bool)
  sub_ind = np.where(sub_ind_bool)[0]
  N_subset = len(sub_ind)
  print('sigma', sigma)
  N_subset = np.round( N_subset + np.random.normal(scale=sigma) )
  N_subset = int( max(N_subset, 0) )

  for i in tqdm(range(n_test_sub)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv_individual = private_tnn_shapley_single_JDP(x_train_few, y_train_few, x_test, y_test, N_subset, tau, K0, sigma*np.sqrt(2), q, dis_metric=dis_metric)
    sv += sv_individual

  return sv, eps, sigma


# x_test, y_test are single data point
def private_knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K, sigma, dis_metric='cosine'):
  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few, dis_metric=dis_metric)
  sv[int(rank[-1])] += int(y_test==y_train_few[int(rank[-1])]) / N + np.random.normal(scale=sigma)

  for j in range(2, N+1):
    i = N+1-j
    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + ( (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / K ) * min(K, i) / i + np.random.normal(scale=sigma)

  return sv

def private_knn_shapley_RJ(x_train_few, y_train_few, x_val_few, y_val_few, K, sigma=0, q=1, delta=1e-5, q_test=1, dis_metric='cosine', rdp=False, eps=-1):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  n_test_sub = int(n_test*q_test)
  test_ind = np.random.choice(range(n_test), size=n_test_sub, replace=False)
  x_val_few, y_val_few = x_val_few[test_ind], y_val_few[test_ind]
  n_compose = n_test_sub

  # If eps is specified, find sigma
  if eps>0 and sigma<0:
    sigma = find_noise_multiplier(sampling_probability=q, num_steps=n_compose, target_epsilon=eps, target_delta=delta, eps_error=1e-3, mu_max=5000)
    sigma = sigma / (K*(K+1))
    print('sigma={}'.format(sigma))
  elif eps<0:
    mech = PrivateKNN_SV_RJ_mech(1, sigma, n_compose, K)
    eps = mech.get_approxDP(delta=delta)

    if rdp or eps < 0.01 or eps > 30:
      mech = PrivateKNN_SV_RJ_mech(1, sigma, n_compose, K)
      eps = mech.get_approxDP(delta=delta)
    else:
      prv = PoissonSubsampledGaussianMechanism(sampling_probability=1, noise_multiplier=sigma * (K*(K+1)) )
      acct = PRVAccountant(prvs=prv, max_self_compositions=n_compose, eps_error=1e-3, delta_error=1e-10)
      low, est, upp = acct.compute_epsilon(delta=delta, num_self_compositions=[n_compose])
      eps = upp
  else:
    pass

  for i in tqdm(range(n_test_sub)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += private_knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K, sigma, dis_metric=dis_metric)

  print(sv)
  print(np.argsort(sv))

  return sv, eps, sigma



# x_test, y_test are single data point
def private_knn_shapley_RJ_withsub_single(x_train_few, y_train_few, x_test, y_test, K, sigma, q, dis_metric='cosine'):

  N = len(y_train_few)
  sv = np.zeros(N)

  for l in range(N):

    # Poisson Subsampling
    sub_ind_bool = (np.random.choice([0, 1], size=N, p=[1-q, q])).astype(bool)
    sub_ind_bool[l] = True
    sub_ind = np.where(sub_ind_bool)[0]

    x_train_few_sub, y_train_few_sub = x_train_few[sub_ind], y_train_few[sub_ind]

    N_sub = len(sub_ind)
    sv_temp = np.zeros(N_sub)

    rank = rank_neighbor(x_test, x_train_few_sub, dis_metric=dis_metric)

    sv_temp[int(rank[-1])] += int(y_test==y_train_few_sub[int(rank[-1])]) / N_sub

    for j in range(2, N_sub+1):
      i = N_sub+1-j
      sv_temp[int(rank[-j])] = sv_temp[int(rank[-(j-1)])] + ( (int(y_test==y_train_few_sub[int(rank[-j])]) - int(y_test==y_train_few_sub[int(rank[-(j-1)])])) / K ) * min(K, i) / i
      if sub_ind[ rank[-j] ] == l:
        break
    sv[l] = sv_temp[int(rank[-j])] + np.random.normal(scale=sigma)

  return sv


def private_knn_shapley_RJ_withsub(x_train_few, y_train_few, x_val_few, y_val_few, K, sigma=0, q=1, delta=1e-5, q_test=1, dis_metric='cosine', rdp=False, eps=-1):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  n_test_sub = int(n_test*q_test)
  test_ind = np.random.choice(range(n_test), size=n_test_sub, replace=False)
  x_val_few, y_val_few = x_val_few[test_ind], y_val_few[test_ind]

  n_compose = n_test_sub

  # If eps is specified, find sigma
  if eps>0 and sigma<0:
    sigma = find_noise_multiplier(sampling_probability=q, num_steps=n_compose, target_epsilon=eps, target_delta=delta, eps_error=1e-2, mu_max=5000)
    sigma = sigma / (K*(K+1))
    print('sigma={}'.format(sigma))
  elif eps<0:
    mech = PrivateKNN_SV_RJ_mech(q, sigma, n_compose, K)
    eps = mech.get_approxDP(delta=delta)

    if rdp or eps < 0.01 or eps > 30:
      mech = PrivateKNN_SV_RJ_mech(q, sigma, n_compose, K)
      eps = mech.get_approxDP(delta=delta)
    else:
      prv = PoissonSubsampledGaussianMechanism(sampling_probability=q, noise_multiplier=sigma * (K*(K+1)) )
      acct = PRVAccountant(prvs=prv, max_self_compositions=n_compose, eps_error=1e-3, delta_error=1e-10)
      low, est, upp = acct.compute_epsilon(delta=delta, num_self_compositions=[n_compose])
      eps = upp
  else:
    pass

  for i in tqdm(range(n_test_sub)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += private_knn_shapley_RJ_withsub_single(x_train_few, y_train_few, x_test, y_test, K, sigma, q=q, dis_metric=dis_metric)

  return sv, eps, sigma







# x_test, y_test are single data point
def knn_banzhaf_single(x_train_few, y_train_few, x_test, y_test, tau=0, K0=10, dis_metric='cosine'):
  N = len(y_train_few)
  sv = np.zeros(N)

  C = max(y_train_few)+1
  if dis_metric == 'cosine':
    # smaller cosine value indicate larger dis-similarity
    distance = -np.dot(x_train_few, x_test) / np.linalg.norm( x_train_few, axis=1 )
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])

  rank = np.argsort(distance)
  if tau == 0:
    Itau = rank[:K0]
  else:
    Itau = (distance<tau).nonzero()[0]

  Ct = len(Itau)
  Ca = np.sum( y_train_few[Itau] == y_test )
  # print('Itau: {}, Ct={}, Ca={}, C={}'.format(Itau, Ct, Ca, C))
  # print(distance[rank[:10]])

  for i in range(N):
    if i in Itau:
      sv[i] = int(y_test==y_train_few[i]) * (2-2**(1-Ct))/Ct - 2**(1-Ct)/C
      if Ct >= 2:
        sv[i] += - (Ca-int(y_test==y_train_few[i])) * (2-(Ct+1)*2**(1-Ct)) / (Ct*(Ct-1))
    else:
      sv[i] = 0

  return sv

def knn_banzhaf(x_train_few, y_train_few, x_val_few, y_val_few, tau=0, K0=10):
  
  N = len(y_train_few)
  sv = np.zeros(N)
  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += knn_banzhaf_single(x_train_few, y_train_few, x_test, y_test, tau, K0)

  return sv




# x_test, y_test are single data point
def private_knn_banzhaf_single(x_train_few, y_train_few, x_test, y_test, tau=0, K0=10, sigma=0, q=1, dis_metric='cosine'):

  N = len(y_train_few)
  sv = np.zeros(N)
  C = max(y_train_few)+1
  t1 = time.time()
  sub_ind_bool = (np.random.choice([0, 1], size=N, p=[1-q, q])).astype(bool)
  sub_ind = np.where(sub_ind_bool)[0]
  if dis_metric == 'cosine':
    distance = -np.dot(x_train_few, x_test) / np.linalg.norm( x_train_few, axis=1 )
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])
  rank_all = np.argsort(distance)
  t2 = time.time()
  if tau == 0:
    tau = x_train_few[rank_all[K0-1]]
  Itau_all = (distance <= tau).nonzero()[0]

  # Itau_subset: index in terms of subset
  distance_subset = distance[sub_ind]
  Itau_subset = (distance_subset <= tau).nonzero()[0]

  Ct = len(Itau_subset) + np.random.normal(scale=sigma)
  Ca = np.sum( y_train_few[sub_ind[Itau_subset]] == y_test ) + np.random.normal(scale=sigma)
  Ct, Ca = np.round(Ct), np.round(Ca)
  Ct, Ca = max(Ct, 0), max(Ca, 0)

  t = time.time()
  print(f'total time of single call is {t - t1}, time to calculate distance is {t2-t1}')
  for i in range(N):

    if i in Itau_all:

      if i in sub_ind:
        Ct_i = max(1, Ct)
        Ca_i = Ca
        if y_test==y_train_few[i]:
          Ca_i = max(1, Ca_i)
      else:
        Ct_i = Ct + 1
        Ct_i = max(1, Ct_i)
        Ca_i = Ca + int(y_test==y_train_few[i])

      sv[i] = int(y_test==y_train_few[i]) * (2-2**(1-Ct_i))/Ct_i - 2**(1-Ct_i)/C
      if Ct_i >= 2:
        sv[i] += - (Ca_i-int(y_test==y_train_few[i])) * (2-(Ct_i+1)*2**(1-Ct_i)) / (Ct_i*(Ct_i-1))

  return sv, (distance<=tau).astype(int)


def private_knn_banzhaf(x_train_few, y_train_few, x_val_few, y_val_few, tau=0, K0=10, sigma=0, q=1, delta=1e-5, q_test=0.1):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  n_test_sub = int(n_test*q_test)
  test_ind = np.random.choice(range(n_test), size=n_test_sub, replace=False)
  x_val_few, y_val_few = x_val_few[test_ind], y_val_few[test_ind]

  n_iter_lst = np.zeros(N)


  for i in range(n_test_sub):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv_individual, close_lst = private_knn_banzhaf_single(x_train_few, y_train_few, x_test, y_test, tau, K0, sigma, q)
    sv += sv_individual
    n_iter_lst += close_lst

  """
  eps_lst = np.zeros(N)
  for i, n_compose in enumerate(n_iter_lst):
    if n_compose == 0:
      eps_lst[i] = 0
    else:
      mech = PrivateKNN_mech(q, sigma, n_compose)
      eps = mech.get_approxDP(delta=delta)
      eps_lst[i] = eps
  """

  print(n_iter_lst)

  n_compose = np.round( np.mean(n_iter_lst) ).astype(int)
  mech = PrivateKNN_mech(q, sigma, n_compose)
  eps = mech.get_approxDP(delta=delta)

  return sv, eps




def get_wtnn_acc(x_train, y_train, x_val, y_val, tau, dis_metric='cosine', kernel='rbf', gamma=1):
  n_val = len(y_val)
  C = max(y_train)+1
  acc = 0
  for i in range(n_val):
    x_test, y_test = x_val[i], y_val[i]
    #ix_test = x_test.reshape((-1,1))
    if dis_metric == 'cosine':
      distance = - np.dot(x_train, x_test) / np.linalg.norm(x_train, axis=1)
    else:
      distance = np.array([np.linalg.norm(x - x_test) for x in x_train])

    Itau = (distance<tau).nonzero()[0]
    acc_single = 0

    if len(Itau) > 0:

      # Only need to consider distance < tau
      distance_tau = distance[Itau]

      if kernel=='rbf':
        weight = np.exp(-(distance_tau+1)*gamma)
      elif kernel=='plain':
        weight = -distance_tau
      else:
        exit(1)

      if max(weight) - min(weight) > 0:
        weight = (weight - min(weight)) / (max(weight) - min(weight))
      weight = weight * ( 2*(y_train[Itau]==y_test)-1 )

      n_digit = 1
      weight_disc = np.round(weight, n_digit)

      if np.sum(weight_disc) > 0:
        acc_single = 1

    else:
      acc_single = 1/C

    acc += acc_single
  return acc / n_val





# x_test, y_test are single data point
def weighted_tknn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau, dis_metric='cosine', kernel='rbf', debug=True):

  N = len(y_train_few)
  sv = np.zeros(N)

  C = max(y_train_few)+1

  # Currently only work for binary classification
  assert C==2

  if dis_metric == 'cosine':
    distance = -np.dot(x_train_few, x_test) / np.linalg.norm(x_train_few, axis=1)
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])

  Itau = (distance < tau).nonzero()[0]
  Ct = len(Itau)
  Ca = np.sum( y_train_few[Itau] == y_test )

  if Ct==0:
    return sv

  # Only need to consider distance < tau
  distance_tau = distance[Itau]

  if debug: print('Ct={}, Ca={}'.format(Ct, Ca))

  if kernel=='rbf':
    gamma = 5
    weight = np.exp(-(distance_tau+1)*gamma)
  elif kernel=='plain':
    weight = -distance_tau
  elif kernel=='uniform':
    weight = np.ones(len(distance_tau))
  else:
    exit(1)

  if max(weight) - min(weight) > 0:
    weight = (weight - min(weight)) / (max(weight) - min(weight))
  weight = weight * ( 2*(y_train_few[Itau]==y_test)-1 )
  if debug: print('weight={}'.format(weight))

  n_digit = 1
  interval = 10**(-n_digit)
  weight_disc = np.round(weight, n_digit)

  # maximum possible range of weights
  weight_max_disc = np.round(np.sum(weight_disc[weight_disc>0]), n_digit)
  weight_min_disc = np.round(np.sum(weight_disc[weight_disc<0]), n_digit)

  N_possible = int(np.round( (weight_max_disc-weight_min_disc)/interval )) + 1
  all_possible = np.linspace(weight_min_disc, weight_max_disc, N_possible)
  if debug: 
    print('weight_max_disc', weight_max_disc)
    print('weight_min_disc', weight_min_disc)
    print('all_possible', all_possible)

  val_ind_map = {}
  for j, val in enumerate(all_possible):
    val_ind_map[np.round(val, n_digit)] = j


  index_zero = val_ind_map[0]
  # print('index of zero: {}'.format(index_zero))
  # print(all_possible[index_zero])

  sv_cache = {}

  big_comb = np.zeros(Ct)
  for l in np.arange(0, Ct, 1):
    big_comb[l] = np.sum([ comb(N-Ct, k-l) / comb(N-1, k) for k in range(l, N) ])

  if debug: 
    print('big_comb={}'.format(big_comb))

  

  # Dynamic Programming
  # i is the index in Itau
  for count, i in tqdm(enumerate(Itau)):

    wi = weight_disc[count]
    weight_loo = np.delete(weight_disc, count)

    if wi > 0:
      check_range = np.round(np.linspace(-wi, 0, int(np.round(wi/interval))+1), n_digit)
    else:
      check_range = np.round(np.linspace(0, -wi, int(np.round(-wi/interval))+1), n_digit)
    if debug: print('Check range of {}th data point: {}'.format(i, check_range))

    if wi in sv_cache:
      sv[i] = sv_cache[wi]
      # print('*** Reuse Shapley Value ***')
      if wi == 0:
        sv[i] = (2*(y_train_few[i]==y_test)-1) * np.abs(sv[i])
    else:
      # A[m, l, s]: number of subsets of size l that uses the first m items in Itau s.t. the sum of weights is s.
      A = np.zeros( (Ct, Ct, len(all_possible)) )

      # Base case: when l=0
      for m in range(Ct):
        A[m, 0, index_zero] = 1

      # for larger l
      for m in range(1, Ct):
        for l in range(1, m+1):
          for j, s in enumerate(all_possible):
            wm = weight_loo[m-1]
            check_val = np.round(s-wm, n_digit)
            if check_val < weight_min_disc or check_val > weight_max_disc:
              A[m, l, j] = A[m-1, l, j]
            else:
              index_interest = val_ind_map[check_val]
              A[m, l, j] = A[m-1, l, j] + A[m-1, l-1, index_interest]

      fi = np.zeros(Ct)
      for l in range(0, Ct):
        for s in check_range:
          if s not in val_ind_map:
            pass
          else:
            index_interest = val_ind_map[s]
            fi[l] += A[Ct-1, l, index_interest]

      if debug: print('fi={}'.format(fi))

      sv[i] = np.dot(big_comb, fi)

      if y_train_few[i] != y_test: 
        sv[i] = -sv[i]

      sv_cache[wi] = sv[i]

  if debug: 
    print('weight_disc', weight_disc)
    print(sv)

  return sv


def weighted_tknn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, tau=0, dis_metric='cosine', kernel='rbf', debug=False):
  
  N = len(y_train_few)
  sv = np.zeros(N)
  n_test = len(y_val_few)

  print('tau in tnn shapley', tau)

  for i in tqdm(range(n_test)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += weighted_tknn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau, dis_metric, kernel, debug)

  return sv





# x_test, y_test are single data point
def fastweighted_tknn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau, dis_metric='cosine', kernel='rbf', debug=True):

  N = len(y_train_few)
  sv = np.zeros(N)

  C = max(y_train_few)+1

  # Currently only work for binary classification
  assert C==2

  if dis_metric == 'cosine':
    distance = -np.dot(x_train_few, x_test) / np.linalg.norm(x_train_few, axis=1)
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])

  Itau = (distance < tau).nonzero()[0]
  Ct = len(Itau)
  Ca = np.sum( y_train_few[Itau] == y_test )

  if Ct==0:
    return sv

  # Only need to consider distance < tau
  distance_tau = distance[Itau]

  if debug: print('Ct={}, Ca={}'.format(Ct, Ca))

  if kernel=='rbf':
    gamma = 5
    weight = np.exp(-(distance_tau+1)*gamma)
  elif kernel=='plain':
    weight = -distance_tau
  elif kernel=='uniform':
    weight = np.ones(len(distance_tau))
  else:
    exit(1)

  if max(weight) - min(weight) > 0:
    weight = (weight - min(weight)) / (max(weight) - min(weight))
  weight = weight * ( 2*(y_train_few[Itau]==y_test)-1 )
  if debug: print('weight={}'.format(weight))

  n_digit = 1
  interval = 10**(-n_digit)
  weight_disc = np.round(weight, n_digit)

  # maximum possible range of weights
  weight_max_disc = np.round(np.sum(weight_disc[weight_disc>0]), n_digit)
  weight_min_disc = np.round(np.sum(weight_disc[weight_disc<0]), n_digit)

  N_possible = int(np.round( (weight_max_disc-weight_min_disc)/interval )) + 1
  all_possible = np.linspace(weight_min_disc, weight_max_disc, N_possible)
  if debug: 
    print('weight_max_disc', weight_max_disc)
    print('weight_min_disc', weight_min_disc)
    print('all_possible', all_possible)

  val_ind_map = {}
  for j, val in enumerate(all_possible):
    val_ind_map[np.round(val, n_digit)] = j


  index_zero = val_ind_map[0]
  # print('index of zero: {}'.format(index_zero))
  # print(all_possible[index_zero])

  sv_cache = {}

  big_comb = np.zeros(Ct)
  for l in np.arange(0, Ct, 1):
    big_comb[l] = np.sum([ comb(N-Ct, k-l) / comb(N-1, k) for k in range(l, N) ])

  if debug: 
    print('big_comb={}'.format(big_comb))

  # Dynamic Programming for computing F(l, s)
  # A[m, l, s]: number of subsets of size l that uses the first m items in Itau s.t. the sum of weights is s.
  A = np.zeros( (Ct+1, Ct+1, len(all_possible)), dtype=object )

  # Base case: l=0
  for m in range(Ct+1):
    A[m, 0, index_zero] = 1

  # for larger l
  for m in range(1, Ct+1):
    for l in range(1, m+1):
      for j, s in enumerate(all_possible):
        wm = weight_disc[m-1]
        check_val = np.round(s-wm, n_digit)
        if check_val < weight_min_disc or check_val > weight_max_disc:
          A[m, l, j] = A[m-1, l, j]
        else:
          index_interest = val_ind_map[check_val]
          A[m, l, j] = A[m-1, l, j] + A[m-1, l-1, index_interest]

  # i is the index in Itau
  for count, i in enumerate(Itau):

    wi = weight_disc[count]

    if wi > 0:
      check_range = np.round(np.linspace(-wi, 0, int(np.round(wi/interval))+1), n_digit)
    else:
      check_range = np.round(np.linspace(0, -wi, int(np.round(-wi/interval))+1), n_digit)

    if debug: 
      print('Check range of {}th data point: {}'.format(i, check_range))

    if wi in sv_cache:
      sv[i] = sv_cache[wi]
      # print('*** Reuse Shapley Value ***')
      if wi == 0:
        sv[i] = (2*(y_train_few[i]==y_test)-1) * np.abs(sv[i])
    else:

      print('A.dtype = {}'.format(A.dtype))

      # B[l, s]: number of subsets of size l in Itau s.t. the sum of weights is s.
      B = np.zeros( (Ct, len(all_possible)), dtype=object)
      B[0, index_zero] = 1

      for l in range(1, Ct):
        for j, s in enumerate(all_possible):
          check_val = np.round(s-wi, n_digit)
          if check_val < weight_min_disc-0.05 or check_val > weight_max_disc+0.05:
            B[l, j] = A[Ct, l, j]
          else:
            index_interest = val_ind_map[check_val]
            B[l, j] = A[Ct, l, j] - B[l-1, index_interest]
            if A[Ct, l, j] < B[l-1, index_interest]:
              print('***WARNING: n_data too large! numerical error!')
              exit(1)

      fi = np.zeros(Ct, dtype=object)
      for l in range(0, Ct):
        for s in check_range:
          if s not in val_ind_map:
            pass
          else:
            index_interest = val_ind_map[s]
            fi[l] += B[l, index_interest]

      if debug: print('fi={}'.format(fi))

      sv[i] = np.dot(big_comb, fi)

      if y_train_few[i] != y_test: 
        sv[i] = -sv[i]

      sv_cache[wi] = sv[i]

  if debug: 
    print('weight_disc', weight_disc)
    print(sv)

  return sv



def fastweighted_tknn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, tau=0, dis_metric='cosine', kernel='rbf', debug=False):
  
  N = len(y_train_few)
  sv = np.zeros(N)
  n_test = len(y_val_few)

  print('tau in tnn shapley', tau)

  for i in tqdm(range(n_test)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += fastweighted_tknn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau, dis_metric, kernel, debug)

  return sv


# x_test, y_test are single data point
def weighted_knn_shapley_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric='cosine', kernel='rbf', debug=True):

  N = len(y_train_few)
  sv = np.zeros(N)

  C = max(y_train_few)+1

  # Currently only work for binary classification
  assert C==2

  # Currently only work for K>1
  assert K > 1

  # Compute distance
  if dis_metric == 'cosine':
    distance = -np.dot(x_train_few, x_test) / np.linalg.norm(x_train_few, axis=1)
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])

  # Compute weights
  if kernel=='rbf':
    gamma = 5
    weight = np.exp(-(distance+1)*gamma)
  elif kernel=='plain':
    weight = -distance
  elif kernel=='uniform':
    weight = np.ones(len(distance))
  else:
    exit(1)

  # We normalize each weight to [0, 1]
  if max(weight) - min(weight) > 0:
    weight = (weight - min(weight)) / (max(weight) - min(weight))

  # Give each weight sign
  weight = weight * ( 2*(y_train_few==y_test)-1 )
  if debug: print('weight={}'.format(weight))

  # Discretize weight to 0.1 precision
  n_digit = 1
  interval = 10**(-n_digit)
  weight_disc = np.round(weight, n_digit)

  # rank weight_disc
  rank = np.argsort(distance)
  weight_disc = weight_disc[rank]

  # maximum possible range of weights
  weight_max_disc = np.round(np.sum(weight_disc[weight_disc>0]), n_digit)
  weight_min_disc = np.round(np.sum(weight_disc[weight_disc<0]), n_digit)

  N_possible = int(np.round( (weight_max_disc-weight_min_disc)/interval )) + 1
  all_possible = np.linspace(weight_min_disc, weight_max_disc, N_possible)
  V = len(all_possible)
  if debug: 
    print('weight_max_disc', weight_max_disc)
    print('weight_min_disc', weight_min_disc)
    print('all_possible', all_possible)

  val_ind_map = {}
  for j, val in enumerate(all_possible):
    val_ind_map[np.round(val, n_digit)] = j


  index_zero = val_ind_map[0]
  if debug:
    print('index of zero: {}, value check = {}'.format(index_zero, all_possible[index_zero]))

  sv_cache = {}
  sv_cache[0] = 0

  for i in range(N):
    print('Now compute {}th Shapley value'.format(i))

    wi = weight_disc[i]

    if wi in sv_cache:
      sv[i] = sv_cache[wi]
      print('*** Reuse Shapley Value ***')

    else:

      # set size+1 for each entry for convenience
      Fi = np.zeros((N, N, V))

      # Initialize for l=1
      for m in range(0, N):
        if m != i:
          wm = weight_disc[m]
          ind_m = val_ind_map[wm]
          Fi[m, 1, ind_m] = 1
      
      # For 2 <= l <= K-1
      for l in range(2, K):
        for m in range(l-1, N):
          if i != m:
            for j, s in enumerate(all_possible):
              wm = weight_disc[m]
              check_val = np.round(s-wm, n_digit)
              if check_val < weight_min_disc or check_val > weight_max_disc:
                Fi[m, l, j] = 0
              else:
                ind_sm = val_ind_map[check_val]
                for t in range(m):
                  if t != i: 
                    Fi[m, l, j] += Fi[t, l-1, ind_sm]

      # For K <= l <= N-1
      for l in range(K, N):
        for m in range( max(i+1, l-1), N ):
          for j, s in enumerate(all_possible):
            for t in range(m):
              if t != i: 
                Fi[m, l, j] += Fi[t, K-1, j] * comb(N-m, l-K)

      Gi = np.zeros(N)

      if wi > 0:
        check_range = np.round(np.linspace(-wi, 0, int(np.round(wi/interval))+1), n_digit)
      elif wi < 0:
        check_range = np.round(np.linspace(0, -wi, int(np.round(-wi/interval))+1), n_digit)
      else:
        check_range = []

      for l in range(1, K):
        for m in range(N):
          for s in check_range:
            if s not in val_ind_map:
              pass
            else:
              ind = val_ind_map[s]
              Gi[l] += Fi[m, l, ind]

      for l in range(K, N):
        for m in range(N):

          wm = weight_disc[m]

          if wi > 0 and wm < wi:
            check_range = np.round(np.linspace(-wi, -wm, int(np.round(wi/interval))+1), n_digit)
          elif wi < 0 and wm > wi:
            check_range = np.round(np.linspace(-wm, -wi, int(np.round(-wi/interval))+1), n_digit)
          else:
            check_range = []

          for s in check_range:
            if s not in val_ind_map:
              pass
            else:
              ind = val_ind_map[s]
              Gi[l] += Fi[m, l, ind]
      
      print('i={}, Gi={}'.format(i, Gi))

      sv[i] = np.sum([ Gi[l]/comb(N-1, l) for l in range(1, N) ])
      if wi < 0:
        sv[i] = -sv[i]

  if debug: 
    print('weight_disc', weight_disc)
    print(sv)

  sv_real = np.zeros(N)
  sv_real[rank] = sv
  sv = sv_real

  if debug: 
    print(sv)

  return sv


def weighted_knn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, K, dis_metric='cosine', kernel='rbf', debug=True):
  
  N = len(y_train_few)
  sv = np.zeros(N)
  n_test = len(y_val_few)

  for i in tqdm(range(n_test)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += weighted_knn_shapley_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric, kernel, debug)

  return sv


def compute_dist(x_train_few, x_test, dis_metric):
  if dis_metric == 'cosine':
    distance = -np.dot(x_train_few, x_test) / np.linalg.norm(x_train_few, axis=1)
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])
  return distance


# temp: temperature coefficient
def compute_weights(distance, kernel, temp=0.9):
  
  distance /= max(distance)

  if kernel=='rbf':
    weight = np.exp(-(distance) / temp)
  elif kernel=='plain':
    weight = -distance
  elif kernel=='uniform':
    weight = np.ones(len(distance))
  else:
    exit(1)

  return weight


def weighted_knn_classification_error(x_train, y_train, x_test, y_test, K, dis_metric, kernel='plain'):
    
    predictions = []
    
    for test_point in x_test:

        distances = compute_dist(x_train, test_point, dis_metric)
        weights = compute_weights(distances, kernel=kernel)

        # We normalize each weight to [0, 1]
        if max(weights) - min(weights) > 0:
          weights = (weights - min(weights)) / (max(weights) - min(weights))

        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:K]
        k_nearest_labels = y_train[k_nearest_indices]
        weights = weights[k_nearest_indices]

        # Voting mechanism
        unique_labels = np.unique(k_nearest_labels)
        weight_sum = [np.sum(weights[k_nearest_labels == label]) for label in unique_labels]
        predicted_label = unique_labels[np.argmax(weight_sum)]
        predictions.append(predicted_label)

    # Calculate classification error
    incorrect_predictions = np.sum(predictions != y_test)
    classification_error = incorrect_predictions / len(y_test)

    return classification_error


def adjust_weights(weight, y_train_few, y_test, y_consider):
    
    # Sanity check
    assert y_consider >= 0
    assert y_consider != y_test

    adjusted_weights = np.zeros_like(weight)
    adjusted_weights[y_train_few == y_test] = weight[y_train_few == y_test]
    adjusted_weights[y_train_few == y_consider] = -weight[y_train_few == y_consider]
    return adjusted_weights.astype(np.float64)


def get_range(weight_disc, n_digit, interval, K):

  sort_pos = np.sort(weight_disc[weight_disc>0])[::-1]
  sort_neg = np.sort(weight_disc[weight_disc<0])

  weight_max_disc = np.round(np.sum(sort_pos[:min(K, len(sort_pos))]), n_digit)
  weight_min_disc = np.round(np.sum(sort_neg[:min(K, len(sort_neg))]), n_digit)

  N_possible = int(np.round( (weight_max_disc-weight_min_disc)/interval )) + 1
  all_possible = np.linspace(weight_min_disc, weight_max_disc, N_possible)
  V = len(all_possible)

  return weight_max_disc, weight_min_disc, N_possible, all_possible, V


def normalize_weight(weight, method='dividemax'):
  if method == 'zeroone':
    if max(weight) - min(weight) > 0:
      weight = (weight - min(weight)) / (max(weight) - min(weight))
  elif method == 'dividemax':
    weight = weight / max(weight)
  return weight


def prepare_weights(x_train_few, y_train_few, x_test, y_test, 
                    dis_metric='cosine', kernel='rbf', y_consider=None, temp=1):

  C = max(y_train_few)+1

  # Compute distance
  distance = compute_dist(x_train_few, x_test, dis_metric)

  # Compute weights
  weight = compute_weights(distance, kernel, temp=temp)

  # We normalize each weight to [0, 1]
  weight = normalize_weight(weight)

  # Adjust weights to give sign
  if C == 2:
    weight = weight * ( 2*(y_train_few==y_test)-1 )
  else:
    weight = adjust_weights(weight, y_train_few, y_test, y_consider)

  return weight, distance





# eps: precision
def fastweighted_knn_shapley_single(weight, distance, K, eps=0, debug=True):

  N = len(distance)
  sv = np.zeros(N)

  if debug: print('weight={}'.format(weight))

  # Discretize weight to 0.1 precision
  n_digit = 1
  interval = 10**(-n_digit)
  weight_disc = np.round(weight, n_digit)

  # reorder weight_disc based on distance rank
  rank = np.argsort(distance)
  weight_disc = weight_disc[rank]

  # maximum possible range of weights  
  weight_max_disc, weight_min_disc, N_possible, all_possible, V = get_range(weight_disc, n_digit, interval, K)
  
  if debug: 
    print('weight_max_disc', weight_max_disc)
    print('weight_min_disc', weight_min_disc)
    print('all_possible', all_possible)
    print('weight_disc', weight_disc)

  val_ind_map = {}
  for j, val in enumerate(all_possible):
    val_ind_map[np.round(val, n_digit)] = j

  index_zero = val_ind_map[0]
  if debug: print('index of zero: {}, value check = {}'.format(index_zero, all_possible[index_zero]))


  # error bound; TODO: improve the efficiency
  def E(mstar):
    mstar += 1
    assert mstar >= K
    A = np.sum( [ 1/(m-K) - 1/m for m in range(mstar+1, N+1) ] )
    B = np.sum( [ (comb(N, l) - comb(mstar, l)) / comb(N-1, l) for l in range(1, K) ] ) / N
    return A + B

  t_Ei = time.time()

  # Compute the smallest m_star s.t. E(m_star) <= eps
  err = 0
  if eps > 0:
    m_star = N-1
    while err < eps and m_star+1 >= K:
      m_star -= 1
      err = E(m_star)
  else:
    m_star = N-1

  t_Ei = time.time()-t_Ei

  if debug:
    print('m_star = {}'.format(m_star))


  t_F = time.time()

  # set size+1 for each entry for convenience
  F = np.zeros((m_star+1, N+1, V))

  # Initialize for l=1
  for m in range(0, m_star+1):
    wm = weight_disc[m]
    ind_m = val_ind_map[wm]
    F[m, 1, ind_m] = 1
      
  # For 2 <= l <= K-1
  for l in range(2, K):
    for m in range(l-1, m_star+1):

      wm = weight_disc[m]
      check_vals = np.round(all_possible - wm, n_digit)

      for j, s in enumerate(all_possible):
        check_val = check_vals[j]
        if check_val < weight_min_disc or check_val > weight_max_disc:
          F[m, l, j] = 0
        else:
          ind_sm = val_ind_map[check_val]
          F[m, l, j] += np.sum(F[:m, l-1, ind_sm])

  if debug:
    print('Computed F; Time: {}'.format(time.time()-t_F))

  t_smallloop = 0
  t_computeGi = 0
  
  I = np.array([ comb(N-1, l) for l in range(0, N) ])

  comb_values = 1.0 / np.array([comb(m, K) for m in range(0, N)])
  deno_values = np.arange(0, N) + 1

  t_Fi = time.time()

  Fi = np.zeros((m_star+1, N, V))

  t_Fi_largerloop = 0

  sv_cache = {}
  sv_cache[0] = 0

  for i in range(N):

    wi = weight_disc[i]

    if wi in sv_cache:
      sv[i] = sv_cache[wi]

    else:

      t_Fi_start = time.time()

      # set size+1 for each entry for convenience
      Fi[:, :, :] = 0

      # Initialize for l=1
      Fi[:, 1, :] = F[:, 1, :]

      if i <= m_star:
        Fi[i, 1, :] = 0

      check_vals = np.round(all_possible-wi, n_digit)
      valid_indices = np.logical_and(check_vals >= weight_min_disc, check_vals <= weight_max_disc)
      invalid_indices = ~valid_indices
      mapped_inds = np.array([val_ind_map[val] for val in check_vals[valid_indices]])

      # For 2 <= l <= K-1
      for l in range(2, K):
        Fi[l-1:i, l, :] = F[l-1:i, l, :]

      for l in range(2, K):
        Fi[max(l-1, i+1):(m_star+1), l, valid_indices] = F[max(l-1, i+1):(m_star+1), l, valid_indices] - Fi[max(l-1, i+1):(m_star+1), l-1, mapped_inds]
        Fi[max(l-1, i+1):(m_star+1), l, invalid_indices] = F[max(l-1, i+1):(m_star+1), l, invalid_indices]
      
      if debug:  
        t_smallloop += time.time()-t_Fi_start
        # print('i={}, small_loop={}'.format(i, time.time()-t_Fi_start))

      t_Gi = time.time()

      Gi = np.zeros(N)

      start_ind, end_ind = 0, -1
      if wi > 0: 
        start_val, end_val = max(-wi, weight_min_disc), -interval
        start_ind, end_ind = val_ind_map[round(start_val, n_digit)], val_ind_map[round(end_val, n_digit)]
      elif wi < 0:
        start_val, end_val = 0, min(-wi-interval, weight_max_disc)
        start_ind, end_ind = val_ind_map[round(start_val, n_digit)], val_ind_map[round(end_val, n_digit)]

      for m in range(m_star+1):
        if i != m:
          Gi[1:K] += np.sum(Fi[m, 1:K, start_ind:end_ind+1], axis=1)

      # Precompute Ri
      Ri = np.zeros(N)

      if wi > 0:
        # start_val, end_val = -wi, wi-interval
        start_val, end_val = max(-wi, weight_min_disc), wi-interval
      elif wi < 0:
        # start_val, end_val = wi, -wi-interval
        start_val, end_val = wi, min(-wi-interval, weight_max_disc)

      start_ind, end_ind = val_ind_map[round(start_val, n_digit)], val_ind_map[round(end_val, n_digit)] 
      R0 = np.sum( Fi[:max(i+1, K), K-1, start_ind:end_ind+1], axis=0 )

      for m in range(max(i+1, K), m_star+1):
        wm = weight_disc[m]
        _, end_ind_m = 0, -1
        if wi > 0 and wm < wi:

          end_val = max(-wm-interval, weight_min_disc)
          end_val = min(end_val, weight_max_disc)

          end_ind_m = val_ind_map[round(end_val, n_digit)]
          Ri[m] = np.sum(R0[:end_ind_m+1-start_ind])
        elif wi < 0 and wm > wi:
          
          start_val = max(-wm, weight_min_disc)
          start_val = min(start_val, weight_max_disc)

          start_ind_m = val_ind_map[round(start_val, n_digit)]
          Ri[m] = np.sum(R0[start_ind_m-start_ind:])
        R0 += Fi[m, K-1, start_ind:end_ind+1]

      ### Test New Approximation Algorithm
      # for m in range(m_star+1, N):
      #   wm = weight_disc[m]
      #   _, end_ind_m = 0, -1
      #   if wi > 0 and wm < wi:

      #     end_val = max(-wm-interval, weight_min_disc)
      #     end_val = min(end_val, weight_max_disc)

      #     end_ind_m = val_ind_map[round(end_val, n_digit)]
      #     Ri[m] = np.sum(R0[:end_ind_m+1-start_ind])
      #   elif wi < 0 and wm > wi:
          
      #     start_val = max(-wm, weight_min_disc)
      #     start_val = min(start_val, weight_max_disc)

      #     start_ind_m = val_ind_map[round(start_val, n_digit)]
      #     Ri[m] = np.sum(R0[start_ind_m-start_ind:])

      if debug:
        print('Ri={}'.format(Ri))

      # sv[i] = np.sum( Ri[max(i+1, K):(m_star+1)] * comb_values[max(i+1, K):(m_star+1)] * N / deno_values[max(i+1, K):(m_star+1)] )
      sv[i] = np.sum( Ri[max(i+1, K):] * comb_values[max(i+1, K):] * N / deno_values[max(i+1, K):] )
      sv[i] += np.sum( Gi[1:K] / I[1:K] )

      if wi < 0:
        sv[i] += 1 # for l=0
        sv[i] = -sv[i]

      t_computeGi += (time.time() - t_Gi)

      t_Fi_largerloop += (time.time() - t_Fi_start)

  print('Computed Fi; Time: {}, Ei_time={}, SmallLoop={}, t_computeGi={}, t_Fi_largerloop={}'.format(
    time.time()-t_Fi, t_Ei, t_smallloop, t_computeGi, t_Fi_largerloop))

  if debug: 
    print('weight_disc', weight_disc)
    print(sv)

  sv_real = np.zeros(N)
  sv_real[rank] = sv
  sv = sv_real

  print('Sanity check: sum of SV = {}, U(N)-U(empty)={}'.format(
      np.sum(sv) / N, int(np.sum(weight_disc[:K]) >= 0)-1 )
  )

  sv = sv / N
  if debug: 
    print(sv)

  return sv







def quantize(value, n_bits):
    """
    Discretizes a real number (or a numpy array of real numbers) between 0 and 1 into n_bits
    and returns its integer representation.

    :param value: Real number or numpy array of real numbers to be quantized (between 0 and 1)
    :param n_bits: Number of bits for discretization
    :return: Integer representation of the quantized value
    """
    n_values = 2**n_bits
    quantized_value = np.round(value * (n_values - 1))

    return quantized_value.astype(int)


def quantize_to_real(value, n_bits):
    """
    Discretizes a real number (or a numpy array of real numbers) between 0 and 1 into n_bits
    and returns its discretized real number representation.

    :param value: Real number or numpy array of real numbers to be quantized (between 0 and 1)
    :param n_bits: Number of bits for discretization
    :return: Real number representation of the quantized value
    """
    n_values = 2**n_bits
    quantized_value = np.round(value * (n_values - 1))

    # Convert the quantized value back to the range [0, 1]
    discretized_real_value = quantized_value / (n_values - 1)

    return discretized_real_value



def get_range_binary(weight_disc, K):
    sort_pos = np.sort(weight_disc[weight_disc > 0])[::-1]
    sort_neg = np.sort(weight_disc[weight_disc < 0])

    weight_max_disc = round(sum(sort_pos[:min(K, len(sort_pos))]))
    weight_min_disc = round(sum(sort_neg[:min(K, len(sort_neg))]))

    # Use range to get all integers between weight_min_disc and weight_max_disc inclusive
    all_possible = np.array(list(range(weight_min_disc, weight_max_disc + 1)))

    N_possible = len(all_possible)

    return weight_max_disc, weight_min_disc, N_possible, all_possible


# Find which endpoint x is closer to
def closest_endpoint(x, weight_min_disc, weight_max_disc):
    if weight_min_disc <= x <= weight_max_disc:
        return x

    if abs(x - weight_min_disc) < abs(x - weight_max_disc):
        return weight_min_disc
    else:
        return weight_max_disc


# eps: precision
# n_bits: number of bit representation
def fastweighted_knn_shapley_binary_single(weights, distance, K, eps=0, debug=True, n_bits=3):

  if debug:
    print('Original Weights={}'.format(weights))

  N = len(distance)
  sv = np.zeros(N)

  # Weights Discretization
  # Note: weight_disc are integers
  weight_disc = quantize(weights, n_bits)

  # reorder weight_disc based on distance rank
  rank = np.argsort(distance)
  weight_disc = weight_disc[rank]

  # maximum possible range of weights
  weight_max_disc, weight_min_disc, V, all_possible = get_range_binary(weight_disc, K)

  if debug:
    print('weight_disc', weight_disc)
    print('weight_max_disc', weight_max_disc)
    print('weight_min_disc', weight_min_disc)
    print('all_possible', all_possible)
    print('weight_disc', weight_disc)

  val_ind_map = {}
  for j, val in enumerate(all_possible):
    val_ind_map[val] = j

  index_zero = val_ind_map[0]

  if debug:
    print('index of zero: {}, value check = {}'.format(index_zero, all_possible[index_zero]))

  # error bound; TODO: improve the efficiency
  def E(mstar):
    mstar += 1
    assert mstar >= K
    A = np.sum( [ 1/(m-K) - 1/m for m in range(mstar+1, N+1) ] )
    B = np.sum( [ (comb(N, l) - comb(mstar, l)) / comb(N-1, l) for l in range(1, K) ] ) / N
    return A + B

  t_Ei = time.time()

  # Compute the smallest m_star s.t. E(m_star) <= eps
  err = 0
  if eps > 0 and N > K:
    m_star = N-1
    while err < eps and m_star+1 >= K:
      m_star -= 1
      err = E(m_star)
  else:
    m_star = N-1

  t_Ei = time.time()-t_Ei

  print('m_star = {}'.format(m_star))

  t_F = time.time()

  # set size+1 for each entry for convenience
  F = np.zeros((m_star+1, N+1, V))

  # Initialize for l=1
  for m in range(0, m_star+1):
    wm = weight_disc[m]
    ind_m = val_ind_map[wm]
    F[m, 1, ind_m] = 1

  # For 2 <= l <= K-1
  for l in range(2, K):
    for m in range(l-1, m_star+1):

      wm = weight_disc[m]
      check_vals = all_possible - wm

      for j, s in enumerate(all_possible):
        check_val = check_vals[j]
        if check_val < weight_min_disc or check_val > weight_max_disc:
          F[m, l, j] = 0
        else:
          ind_sm = val_ind_map[check_val]
          F[m, l, j] += np.sum(F[:m, l-1, ind_sm])

  if debug:
    print('Computed F; Time: {}'.format(time.time()-t_F))

  t_smallloop = 0
  t_computeGi = 0

  I = np.array([ comb(N-1, l) for l in range(0, N) ])

  comb_values = np.zeros(N)
  comb_values[K:] = 1.0 / np.array([comb(m, K) for m in range(K, N)])
  deno_values = np.arange(0, N) + 1

  t_Fi = time.time()

  Fi = np.zeros((m_star+1, N, V))

  t_Fi_largerloop = 0

  sv_cache = {}
  sv_cache[0] = 0

  for i in range(N):

    wi = weight_disc[i]

    if wi in sv_cache:
      sv[i] = sv_cache[wi]

    else:

      t_Fi_start = time.time()

      # set size+1 for each entry for convenience
      Fi[:, :, :] = 0

      # Initialize for l=1
      Fi[:, 1, :] = F[:, 1, :]

      if i <= m_star:
        Fi[i, 1, :] = 0

      check_vals = all_possible-wi
      valid_indices = np.logical_and(check_vals >= weight_min_disc, check_vals <= weight_max_disc)
      invalid_indices = ~valid_indices
      mapped_inds = np.array([val_ind_map[val] for val in check_vals[valid_indices]])

      # For 2 <= l <= K-1
      for l in range(2, K):
        Fi[l-1:i, l, :] = F[l-1:i, l, :]

      for l in range(2, K):
        Fi[max(l-1, i+1):(m_star+1), l, valid_indices] = F[max(l-1, i+1):(m_star+1), l, valid_indices] - Fi[max(l-1, i+1):(m_star+1), l-1, mapped_inds]
        Fi[max(l-1, i+1):(m_star+1), l, invalid_indices] = F[max(l-1, i+1):(m_star+1), l, invalid_indices]

      # if debug:
      #   t_smallloop += time.time()-t_Fi_start
      #   print('i={}, small_loop={}'.format(i, time.time()-t_Fi_start))

      t_Gi = time.time()

      Gi = np.zeros(N)

      if wi > 0:
        start_val, end_val = max(-wi, weight_min_disc), closest_endpoint(-1, weight_min_disc, weight_max_disc)
        start_ind, end_ind = val_ind_map[start_val], val_ind_map[end_val]
      elif wi < 0:
        start_val, end_val = closest_endpoint(0, weight_min_disc, weight_max_disc), min(-wi-1, weight_max_disc)
        start_ind, end_ind = val_ind_map[start_val], val_ind_map[end_val]
      else:
        sys.exit(1)

      for m in range(m_star+1):
        if i != m:
          Gi[1:K] += np.sum(Fi[m, 1:K, start_ind:end_ind+1], axis=1)

      # Precompute Ri
      Ri = np.zeros(N)

      if wi > 0:
        start_val, end_val = max(-wi, weight_min_disc), wi-1
      elif wi < 0:
        start_val, end_val = wi, min(-wi-1, weight_max_disc)
      start_ind, end_ind = val_ind_map[start_val], val_ind_map[end_val]

      R0 = np.sum( Fi[:max(i+1, K), K-1, start_ind:end_ind+1], axis=0 )

      for m in range(max(i+1, K), m_star+1):
        wm = weight_disc[m]
        _, end_ind_m = 0, -1
        if wi > 0 and wm < wi:
          end_val = max(-wm-1, weight_min_disc)
          end_val = min(end_val, weight_max_disc)
          end_ind_m = val_ind_map[end_val]
          Ri[m] = np.sum(R0[:end_ind_m+1-start_ind])
        elif wi < 0 and wm > wi:
          start_val = max(-wm, weight_min_disc)
          start_val = min(start_val, weight_max_disc)
          start_ind_m = val_ind_map[start_val]
          Ri[m] = np.sum(R0[start_ind_m-start_ind:])

        R0 += Fi[m, K-1, start_ind:end_ind+1]

      if debug:
        print('Ri={}'.format(Ri))

      sv[i] = np.sum( Ri[max(i+1, K):] * comb_values[max(i+1, K):] * N / deno_values[max(i+1, K):] )
      sv[i] += np.sum( Gi[1:K] / I[1:K] )

      if wi < 0:
        sv[i] += 1 # for l=0
        sv[i] = -sv[i]

      t_computeGi += (time.time() - t_Gi)

      t_Fi_largerloop += (time.time() - t_Fi_start)

  print('Computed Fi; Time: {}, Ei_time={}, SmallLoop={}, t_computeGi={}, t_Fi_largerloop={}'.format(
    time.time()-t_Fi, t_Ei, t_smallloop, t_computeGi, t_Fi_largerloop))

  weight_disc_real = quantize_to_real(weights, n_bits)

  sv_real = np.zeros(N)
  sv_real[rank] = sv
  sv = sv_real

  print('Sanity check: sum of SV = {}, U(N)-U(empty)={}'.format(
      np.sum(sv) / N, int(np.sum(weight_disc[:K]) >= 0)-1 )
  )

  sv = sv / N

  if debug:
    print('weights (discretized):', weight_disc_real)
    print(sv)

  return sv




# eps: precision
# n_bits: number of bit representation
def fastweighted_knn_shapley_binary_single_changebase(weights, distance, K, eps=0, debug=True, n_bits=3):

  if debug:
    print('Original Weights={}'.format(weights))

  N = len(distance)
  sv = np.zeros(N)

  # Weights Discretization
  # Note: weight_disc are integers
  weight_disc = quantize(weights, n_bits)

  # reorder weight_disc based on distance rank
  rank = np.argsort(distance)
  weight_disc = weight_disc[rank]

  # maximum possible range of weights
  weight_max_disc, weight_min_disc, V, all_possible = get_range_binary(weight_disc, K)

  if debug:
    print('weight_disc', weight_disc)
    print('weight_max_disc', weight_max_disc)
    print('weight_min_disc', weight_min_disc)
    print('all_possible', all_possible)
    print('weight_disc', weight_disc)

  val_ind_map = {}
  for j, val in enumerate(all_possible):
    val_ind_map[val] = j

  index_zero = val_ind_map[0]

  if debug:
    print('index of zero: {}, value check = {}'.format(index_zero, all_possible[index_zero]))

  # error bound; TODO: improve the efficiency
  def E(mstar):
    mstar += 1
    assert mstar >= K
    A = np.sum( [ 1/(m-K) - 1/m for m in range(mstar+1, N+1) ] )
    B = np.sum( [ (comb(N, l) - comb(mstar, l)) / comb(N-1, l) for l in range(1, K) ] ) / N
    return A + B

  t_Ei = time.time()

  # Compute the smallest m_star s.t. E(m_star) <= eps
  err = 0
  if eps > 0 and N > K:
    m_star = N-1
    while err < eps and m_star+1 >= K:
      m_star -= 1
      err = E(m_star)
  else:
    m_star = N-1

  t_Ei = time.time()-t_Ei

  print('m_star = {}'.format(m_star))

  t_F = time.time()

  # set size+1 for each entry for convenience
  F = np.zeros((m_star+1, N+1, V))

  # Initialize for l=1
  for m in range(0, m_star+1):
    wm = weight_disc[m]
    ind_m = val_ind_map[wm]
    F[m, 1, ind_m] = 1

  # For 2 <= l <= K-1
  for l in range(2, K):
    for m in range(l-1, m_star+1):

      wm = weight_disc[m]
      check_vals = all_possible - wm

      for j, s in enumerate(all_possible):
        check_val = check_vals[j]
        if check_val < weight_min_disc or check_val > weight_max_disc:
          F[m, l, j] = 0
        else:
          ind_sm = val_ind_map[check_val]
          F[m, l, j] += np.sum(F[:m, l-1, ind_sm])

  if debug:
    print('Computed F; Time: {}'.format(time.time()-t_F))

  t_smallloop = 0
  t_computeGi = 0

  I = np.array([ comb(N-1, l) for l in range(0, N) ])

  comb_values = np.zeros(N)
  comb_values[K:] = 1.0 / np.array([comb(m, K) for m in range(K, N)])
  deno_values = np.arange(0, N) + 1

  t_Fi = time.time()

  Fi = np.zeros((m_star+1, N, V))

  t_Fi_largerloop = 0

  sv_cache = {}
  sv_cache[0] = 0

  for i in range(N):

    wi = weight_disc[i]

    if wi in sv_cache:
      sv[i] = sv_cache[wi]

    else:

      t_Fi_start = time.time()

      # set size+1 for each entry for convenience
      Fi[:, :, :] = 0

      # Initialize for l=1
      Fi[:, 1, :] = F[:, 1, :]

      if i <= m_star:
        Fi[i, 1, :] = 0

      check_vals = all_possible-wi
      valid_indices = np.logical_and(check_vals >= weight_min_disc, check_vals <= weight_max_disc)
      invalid_indices = ~valid_indices
      mapped_inds = np.array([val_ind_map[val] for val in check_vals[valid_indices]])

      # For 2 <= l <= K-1
      for l in range(2, K):
        Fi[l-1:i, l, :] = F[l-1:i, l, :]

      for l in range(2, K):
        Fi[max(l-1, i+1):(m_star+1), l, valid_indices] = F[max(l-1, i+1):(m_star+1), l, valid_indices] - Fi[max(l-1, i+1):(m_star+1), l-1, mapped_inds]
        Fi[max(l-1, i+1):(m_star+1), l, invalid_indices] = F[max(l-1, i+1):(m_star+1), l, invalid_indices]

      # if debug:
      #   t_smallloop += time.time()-t_Fi_start
      #   print('i={}, small_loop={}'.format(i, time.time()-t_Fi_start))

      t_Gi = time.time()

      Gi = np.zeros(N)

      if wi > 0:
        start_val, end_val = max(-wi+1, weight_min_disc), closest_endpoint(0, weight_min_disc, weight_max_disc)
        start_ind, end_ind = val_ind_map[start_val], val_ind_map[end_val]
      elif wi < 0:
        start_val, end_val = closest_endpoint(1, weight_min_disc, weight_max_disc), min(-wi, weight_max_disc)
        start_ind, end_ind = val_ind_map[start_val], val_ind_map[end_val]
      else:
        sys.exit(1)

      for m in range(m_star+1):
        if i != m:
          Gi[1:K] += np.sum(Fi[m, 1:K, start_ind:end_ind+1], axis=1)

      # Precompute Ri
      Ri = np.zeros(N)

      if wi > 0:
        start_val, end_val = max(-wi+1, weight_min_disc), wi
      elif wi < 0:
        start_val, end_val = wi+1, min(-wi, weight_max_disc)
      start_ind, end_ind = val_ind_map[start_val], val_ind_map[end_val]

      R0 = np.sum( Fi[:max(i+1, K), K-1, start_ind:end_ind+1], axis=0 )

      for m in range(max(i+1, K), m_star+1):
        wm = weight_disc[m]
        if wi > 0 and wm < wi:
          end_val = max(-wm, weight_min_disc)
          end_val = min(end_val, weight_max_disc)
          end_ind_m = val_ind_map[end_val]
          Ri[m] = np.sum(R0[:end_ind_m+1-start_ind])
        elif wi < 0 and wm > wi:
          start_val = max(-wm+1, weight_min_disc)
          start_val = min(start_val, weight_max_disc)
          start_ind_m = val_ind_map[start_val]
          Ri[m] = np.sum(R0[start_ind_m-start_ind:])

        R0 += Fi[m, K-1, start_ind:end_ind+1]

      if debug:
        print('Ri={}'.format(Ri))

      sv[i] = np.sum( Ri[max(i+1, K):] * comb_values[max(i+1, K):] * N / deno_values[max(i+1, K):] )
      sv[i] += np.sum( Gi[1:K] / I[1:K] )

      if wi < 0:
        sv[i] = -sv[i]
      elif wi > 0:
        sv[i] += 1 # for l=0

      t_computeGi += (time.time() - t_Gi)

      t_Fi_largerloop += (time.time() - t_Fi_start)

  print('Computed Fi; Time: {}, Ei_time={}, SmallLoop={}, t_computeGi={}, t_Fi_largerloop={}'.format(
    time.time()-t_Fi, t_Ei, t_smallloop, t_computeGi, t_Fi_largerloop))

  weight_disc_real = quantize_to_real(weights, n_bits)

  sv_real = np.zeros(N)
  sv_real[rank] = sv
  sv = sv_real

  print('Sanity check: sum of SV = {}, U(N)-U(empty)={}'.format(
      np.sum(sv) / N, int(np.sum(weight_disc[:K]) > 0) )
  )

  sv = sv / N

  if debug:
    print('weights (discretized):', weight_disc_real)
    print(sv)

  return sv








def fastweighted_knn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, K, 
                             eps, dis_metric='cosine', kernel='rbf', debug=False, n_bits=3, collect_sv=False, 
                             temp=1):
  
  # Currently only work for K>1
  assert K > 1

  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  C = max(y_train_few)+1

  print('Number of classes = {}'.format(C))

  distinct_classes = np.arange(C)

  sv_lst = []

  for i in tqdm(range(n_test)):

    x_test, y_test = x_val_few[i], y_val_few[i]

    if C==2:
      weight, distance = prepare_weights(x_train_few, y_train_few, x_test, y_test, dis_metric, kernel, y_consider=None, temp=temp)
      sv_i = fastweighted_knn_shapley_binary_single_changebase(weight, distance, K, eps, debug, n_bits)
    else:
      sv_i = np.zeros(N)
      classes_to_enumerate = distinct_classes[distinct_classes != y_test]
      for c in classes_to_enumerate:
        weight, distance = prepare_weights(x_train_few, y_train_few, x_test, y_test, dis_metric, kernel, y_consider=c, temp=temp)
        nonzero_ind = np.nonzero(weight)[0]
        sv_temp = fastweighted_knn_shapley_binary_single_changebase(weight[nonzero_ind], distance[nonzero_ind], K=min(K, len(nonzero_ind)), eps=eps, debug=debug, n_bits=n_bits)
        sv_i[nonzero_ind] += sv_temp

    sv += sv_i
    sv_lst.append(sv_i)

  if collect_sv:
    return sv, sv_lst
  else:
    return sv









# # x_test, y_test are single data point
# def fastweighted_knn_shapley_old_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric='cosine', kernel='rbf', debug=True):

#   N = len(y_train_few)
#   sv = np.zeros(N)

#   C = max(y_train_few)+1

#   # Currently only work for binary classification
#   assert C==2

#   # Currently only work for K>1
#   assert K > 1

#   # Compute distance
#   distance = compute_dist(x_train_few, x_test, dis_metric)

#   # Compute weights
#   weight = compute_weights(distance, kernel)

#   # We normalize each weight to [0, 1]
#   if max(weight) - min(weight) > 0:
#     weight = (weight - min(weight)) / (max(weight) - min(weight))

#   # Give each weight sign
#   weight = weight * ( 2*(y_train_few==y_test)-1 )
#   if debug: print('weight={}'.format(weight))

#   # Discretize weight to 0.1 precision
#   n_digit = 1
#   interval = 10**(-n_digit)
#   weight_disc = np.round(weight, n_digit)

#   # reorder weight_disc based on rank
#   rank = np.arange(N).astype(int)
#   weight_disc = weight_disc[rank]

#   # maximum possible range of weights  
#   weight_max_disc, weight_min_disc, N_possible, all_possible, V = get_range(weight_disc, n_digit, interval, K)
  
#   if debug: 
#     print('weight_max_disc', weight_max_disc)
#     print('weight_min_disc', weight_min_disc)
#     print('all_possible', all_possible)

#   val_ind_map = {}
#   for j, val in enumerate(all_possible):
#     val_ind_map[np.round(val, n_digit)] = j

#   index_zero = val_ind_map[0]
#   if debug:
#     print('index of zero: {}, value check = {}'.format(index_zero, all_possible[index_zero]))


#   # set size+1 for each entry for convenience
#   F = np.zeros((N, N+1, V))

#   # Initialize for l=1
#   for m in range(0, N):
#     wm = weight_disc[m]
#     ind_m = val_ind_map[wm]
#     F[m, 1, ind_m] = 1
      
#   # For 2 <= l <= K-1
#   for l in range(2, K):
#     for m in range(l-1, N):
#       for j, s in enumerate(all_possible):
#         wm = weight_disc[m]
#         check_val = np.round(s-wm, n_digit)
#         if check_val < weight_min_disc or check_val > weight_max_disc:
#           F[m, l, j] = 0
#         else:
#           ind_sm = val_ind_map[check_val]
#           for t in range(m):
#             F[m, l, j] += F[t, l-1, ind_sm]

#   # For K <= l <= N
#   for l in range(K, N+1):
#     for m in range(l-1, N):
#       for j, s in enumerate(all_possible):
#         for t in range(m):
#           F[m, l, j] += F[t, K-1, j] * comb(N-m, l-K)


#   sv_cache = {}
#   sv_cache[0] = 0

#   for i in range(N):
#     print('Now compute {}th Shapley value'.format(i))

#     wi = weight_disc[i]

#     if wi in sv_cache:
#       sv[i] = sv_cache[wi]
#       print('*** Reuse Shapley Value ***')

#     else:

#       # set size+1 for each entry for convenience
#       Fi = np.zeros((N, N, V))

#       # Initialize for l=1
#       for m in range(0, N):
#         if m != i:
#           wm = weight_disc[m]
#           ind_m = val_ind_map[wm]
#           Fi[m, 1, ind_m] = 1
      
#       # For 2 <= l <= K-1
#       for l in range(2, K):
#         for m in range(l-1, N):
#           for j, s in enumerate(all_possible):
#             if i < m:
#               check_val = np.round(s-wi, n_digit)
#               if check_val < weight_min_disc or check_val > weight_max_disc:
#                 Fi[m, l, j] = F[m, l, j]
#               else:
#                 ind_sm = val_ind_map[check_val]
#                 Fi[m, l, j] = F[m, l, j] - Fi[m, l-1, ind_sm]
#             elif i > m:
#               Fi[m, l, j] = F[m, l, j]

#       # For K <= l <= N-1
#       for l in range(K, N):
#         for m in range( max(i+1, l-1), N ):
#           for j, s in enumerate(all_possible):
#             wm = weight_disc[m]
#             check_val = np.round(s-wi+wm, n_digit)
#             if check_val < weight_min_disc or check_val > weight_max_disc:
#               Fi[m, l, j] = F[m, l, j]
#             else:
#               ind_sm = val_ind_map[check_val]
#               Fi[m, l, j] = F[m, l, j] - Fi[m, K-1, ind_sm] * comb(N-m, l-K)

#       Gi = np.zeros(N)

#       if wi > 0:
#         check_range = np.round(np.linspace(-wi, 0, int(np.round(wi/interval))+1), n_digit)
#       elif wi < 0:
#         check_range = np.round(np.linspace(0, -wi, int(np.round(-wi/interval))+1), n_digit)
#       else:
#         check_range = []

#       for l in range(1, K):
#         for m in range(N):
#           if i != m:
#             for s in check_range:
#               if s not in val_ind_map:
#                 pass
#               else:
#                 ind = val_ind_map[s]
#                 Gi[l] += Fi[m, l, ind]

#       for l in range(K, N):
#         for m in range(N):
#           if i != m:
#             wm = weight_disc[m]
#             if wi > 0 and wm < wi:
#               check_range = np.round(np.linspace(-wi, -wm, int(np.round(wi/interval))+1), n_digit)
#             elif wi < 0 and wm > wi:
#               check_range = np.round(np.linspace(-wm, -wi, int(np.round(-wi/interval))+1), n_digit)
#             else:
#               check_range = []

#             for s in check_range:
#               if s not in val_ind_map:
#                 pass
#               else:
#                 ind = val_ind_map[s]
#                 Gi[l] += Fi[m, l, ind]

#       sv[i] = np.sum([ Gi[l]/comb(N-1, l) for l in range(1, N) ])
#       if wi < 0:
#         sv[i] = -sv[i]

#   if debug: 
#     print('weight_disc', weight_disc)
#     print(sv)

#   sv_real = np.zeros(N)
#   sv_real[rank] = sv
#   sv = sv_real

#   if debug: 
#     print(2*(y_train_few==y_test)-1)
#     print(sv)
#     print(np.sum(sv))

#   return sv



# def fastweighted_knn_shapley_old(x_train_few, y_train_few, x_val_few, y_val_few, K, dis_metric='cosine', kernel='rbf', debug=True):
  
#   N = len(y_train_few)
#   sv = np.zeros(N)
#   n_test = len(y_val_few)

#   for i in tqdm(range(n_test)):
#     x_test, y_test = x_val_few[i], y_val_few[i]
#     sv += fastweighted_knn_shapley_old_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric, kernel, debug)

#   return sv







# # x_test, y_test are single data point
# def approxfastweighted_knn_shapley_single(x_train_few, y_train_few, x_test, y_test, K, eps, dis_metric='cosine', kernel='rbf', debug=True):

#   N = len(y_train_few)
#   sv = np.zeros(N)

#   C = max(y_train_few)+1

#   # Currently only work for binary classification
#   assert C==2

#   # Currently only work for K>1
#   assert K > 1

#   # Compute distance
#   distance = compute_dist(x_train_few, x_test, dis_metric)

#   # Compute weights
#   weight = compute_weights(distance, kernel)

#   # We normalize each weight to [0, 1]
#   if max(weight) - min(weight) > 0:
#     weight = (weight - min(weight)) / (max(weight) - min(weight))

#   # Give each weight sign
#   weight = weight * ( 2*(y_train_few==y_test)-1 )
#   if debug: print('weight={}'.format(weight))

#   # Discretize weight to 0.1 precision
#   n_digit = 1
#   interval = 10**(-n_digit)
#   weight_disc = np.round(weight, n_digit)

#   # reorder weight_disc based on rank
#   rank = np.argsort(distance)
#   weight_disc = weight_disc[rank]

#   # maximum possible range of weights
#   weight_max_disc, weight_min_disc, N_possible, all_possible, V = get_range(weight_disc, n_digit, interval, K)

#   if debug: 
#     print('weight_max_disc', weight_max_disc)
#     print('weight_min_disc', weight_min_disc)
#     print('all_possible', all_possible)

#   val_ind_map = {}
#   for j, val in enumerate(all_possible):
#     val_ind_map[np.round(val, n_digit)] = j

#   index_zero = val_ind_map[0]
#   if debug:
#     print('index of zero: {}, value check = {}'.format(index_zero, all_possible[index_zero]))


#   # set size+1 for each entry for convenience
#   F = np.zeros((N, N+1, V))

#   # Initialize for l=1
#   for m in range(0, N):
#     wm = weight_disc[m]
#     ind_m = val_ind_map[wm]
#     F[m, 1, ind_m] = 1
      
#   # For 2 <= l <= K-1
#   for l in range(2, K):
#     for m in range(l-1, N):
#       for j, s in enumerate(all_possible):
#         wm = weight_disc[m]
#         check_val = np.round(s-wm, n_digit)
#         if check_val < weight_min_disc or check_val > weight_max_disc:
#           F[m, l, j] = 0
#         else:
#           ind_sm = val_ind_map[check_val]
#           for t in range(m):
#             F[m, l, j] += F[t, l-1, ind_sm]

#   # For K <= l <= N
#   for l in range(K, N+1):
#     for m in range(l-1, N):
#       for j, s in enumerate(all_possible):
#         for t in range(m):
#           F[m, l, j] += F[t, K-1, j] * comb(N-m, l-K)

#   # error bound for setting Gi(l)=0; TODO: improve the efficiency of computing E(i, l)
#   def E(i, l):
#     i = i+1
#     return np.sum( [comb(i-1, j)*comb(N-i, l-j) for j in range(K)] / comb(N-1, l) )

#   sv_cache = {}
#   sv_cache[0] = 0

#   for i in range(N):
#     print('Now compute {}th Shapley value'.format(i))

#     wi = weight_disc[i]

#     if wi in sv_cache:
#       sv[i] = sv_cache[wi]
#       print('*** Reuse Shapley Value ***')

#     else:

#       # set size+1 for each entry for convenience
#       Fi = np.zeros((N, N, V))

#       # Initialize for l=1
#       for m in range(0, N):
#         if m != i:
#           wm = weight_disc[m]
#           ind_m = val_ind_map[wm]
#           Fi[m, 1, ind_m] = 1
      
#       # For 2 <= l <= K-1
#       for l in range(2, K):
#         for m in range(l-1, N):
#           for j, s in enumerate(all_possible):
#             if i < m:
#               check_val = np.round(s-wi, n_digit)
#               if check_val < weight_min_disc or check_val > weight_max_disc:
#                 Fi[m, l, j] = F[m, l, j]
#               else:
#                 ind_sm = val_ind_map[check_val]
#                 Fi[m, l, j] = F[m, l, j] - Fi[m, l-1, ind_sm]
#             elif i > m:
#               Fi[m, l, j] = F[m, l, j]

#       # Compute l_star s.t. error <= eps
#       err = 0
#       l_star = N
#       while err < eps:
#         l_star -= 1
#         err += E(i, l_star)

#       print('l_star = {}'.format(l_star))

#       # For K <= l <= N-1
#       for l in range(K, l_star+1):
#         for m in range( max(i+1, l-1), N ):
#           for j, s in enumerate(all_possible):
#             wm = weight_disc[m]
#             check_val = np.round(s-wi+wm, n_digit)
#             if check_val < weight_min_disc or check_val > weight_max_disc:
#               Fi[m, l, j] = F[m, l, j]
#             else:
#               ind_sm = val_ind_map[check_val]
#               Fi[m, l, j] = F[m, l, j] - Fi[m, K-1, ind_sm] * comb(N-m, l-K)

#       Gi = np.zeros(N)

#       if wi > 0:
#         check_range = np.round(np.linspace(-wi, -interval, int(np.round(wi/interval))), n_digit)
#       elif wi < 0:
#         check_range = np.round(np.linspace(0, -wi-interval, int(np.round(-wi/interval))), n_digit)
#       else:
#         check_range = []

#       for l in range(1, K):
#         for m in range(N):
#           if i != m:
#             for s in check_range:
#               if s not in val_ind_map:
#                 pass
#               else:
#                 ind = val_ind_map[s]
#                 Gi[l] += Fi[m, l, ind]

#       for l in range(K, l_star+1):
#         for m in range(N):
#           wm = weight_disc[m]

#           if wi > 0 and wm < wi:
#             check_range = np.round(np.linspace(-wi, -wm-interval, int(np.round((wi-wm)/interval))), n_digit)
#           elif wi < 0 and wm > wi:
#             check_range = np.round(np.linspace(-wm, -wi-interval, int(np.round((wm-wi)/interval))), n_digit)
#           else:
#             check_range = []

#           for s in check_range:
#             if s not in val_ind_map:
#               pass
#             else:
#               ind = val_ind_map[s]
#               Gi[l] += Fi[m, l, ind]

#       sv[i] = np.sum([ Gi[l]/comb(N-1, l) for l in range(1, N) ])
#       if wi < 0:
#         sv[i] = -sv[i]

#   if debug: 
#     print('weight_disc', weight_disc)
#     print(sv)

#   sv_real = np.zeros(N)
#   sv_real[rank] = sv
#   sv = sv_real

#   if debug: 
#     print(2*(y_train_few==y_test)-1)
#     print(sv)
#     print(np.sum(sv))

#   return sv



# def approxfastweighted_knn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, K, eps, dis_metric='cosine', kernel='rbf', debug=True):
  
#   N = len(y_train_few)
#   sv = np.zeros(N)
#   n_test = len(y_val_few)

#   for i in tqdm(range(n_test)):
#     x_test, y_test = x_val_few[i], y_val_few[i]
#     sv += approxfastweighted_knn_shapley_single(x_train_few, y_train_few, x_test, y_test, K, eps, dis_metric, kernel, debug)

#   return sv



