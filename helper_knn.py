import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
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
from os.path import exists
import warnings

from tqdm import tqdm

import scipy
from scipy.special import beta, comb
from random import randint

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from utility_func import *

from helper_privacy import PrivateKNN_mech, PrivateKNN_SV_RJ_mech

import prv_accountant
from prv_accountant.other_accountants import RDP
from prv_accountant import PRVAccountant, PoissonSubsampledGaussianMechanism
from prv_accountant.dpsgd import find_noise_multiplier

import config

from sklearn.metrics.pairwise import cosine_similarity

import gmpy2


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
def knn_shapley_RJ(x_train_few, y_train_few, x_val_few, y_val_few, K, dis_metric='cosine'):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric='cosine')

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


def get_knn_acc(x_train, y_train, x_val, y_val, K, dis_metric='cosine'):
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

def compute_weights(distance, kernel):
  if kernel=='rbf':
    gamma = 5
    weight = np.exp(-(distance+1)*gamma)
  elif kernel=='plain':
    weight = -distance
  elif kernel=='uniform':
    weight = np.ones(len(distance))
  else:
    exit(1)

  return weight

def get_range(weight_disc, n_digit, interval, K):

  sort_pos = np.sort(weight_disc[weight_disc>0])[::-1]
  sort_neg = np.sort(weight_disc[weight_disc<0])

  weight_max_disc = np.round(np.sum(sort_pos[:min(K, len(sort_pos))]), n_digit)
  weight_min_disc = np.round(np.sum(sort_neg[:min(K, len(sort_neg))]), n_digit)

  N_possible = int(np.round( (weight_max_disc-weight_min_disc)/interval )) + 1
  all_possible = np.linspace(weight_min_disc, weight_max_disc, N_possible)
  V = len(all_possible)

  return weight_max_disc, weight_min_disc, N_possible, all_possible, V


# x_test, y_test are single data point
# eps: precision
def fastweighted_knn_shapley_single(x_train_few, y_train_few, x_test, y_test, K, eps=0, dis_metric='cosine', kernel='rbf', debug=True):

  N = len(y_train_few)
  sv = np.zeros(N)

  C = max(y_train_few)+1

  # Currently only work for binary classification
  assert C==2

  # Currently only work for K>1
  assert K > 1

  # Compute distance
  distance = compute_dist(x_train_few, x_test, dis_metric)

  # Compute weights
  weight = compute_weights(distance, kernel)

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

  # reorder weight_disc based on rank
  rank = np.argsort(distance)
  weight_disc = weight_disc[rank]

  # maximum possible range of weights  
  weight_max_disc, weight_min_disc, N_possible, all_possible, V = get_range(weight_disc, n_digit, interval, K)
  
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


  t_F = time.time()

  # set size+1 for each entry for convenience
  F = np.zeros((N, N+1, V))

  # Initialize for l=1
  for m in range(0, N):
    wm = weight_disc[m]
    ind_m = val_ind_map[wm]
    F[m, 1, ind_m] = 1
      
  # For 2 <= l <= K-1
  for l in range(2, K):
    for m in range(l-1, N):

      wm = weight_disc[m]
      check_vals = np.round(all_possible - wm, n_digit)

      for j, s in enumerate(all_possible):
        check_val = check_vals[j]
        if check_val < weight_min_disc or check_val > weight_max_disc:
          F[m, l, j] = 0
        else:
          ind_sm = val_ind_map[check_val]
          F[m, l, j] += np.sum(F[:m, l-1, ind_sm])

  l_values, m_values = np.ogrid[K:N+1, K-1:N]
  H = np.zeros((N+1, N))
  H[K:N+1, K-1:N] = comb(N-1-m_values, l_values-K)

  for j, s in enumerate(all_possible):
    Q = np.sum(F[:(K-1), K-1, j])
    for m in range(K-1, N):
      F[m, K:(N+1), j] = Q * H[K:(N+1), m]
      Q += F[m, K-1, j]

  print('Computed F; Time: {}'.format(time.time()-t_F))

  t_Fi = time.time()
  t_Ei = 0
  t_bigloop = 0
  t_computeGi = 0
  
  I = np.array([ comb(N-1, l) for l in range(0, N) ])

  # error bound for setting Gi(l)=0; TODO: improve the efficiency of computing E(i, l)
  def E(i, l):
    i = i+1
    return np.sum( [comb(i-1, j)*comb(N-i, l-j) for j in range(K)] / comb(N-1, l) )

  sv_cache = {}
  sv_cache[0] = 0

  for i in range(N):
    # print('Now compute {}th Shapley value'.format(i))

    wi = weight_disc[i]

    if wi in sv_cache:
      sv[i] = sv_cache[wi]
      # print('*** Reuse Shapley Value ***')

    else:

      # set size+1 for each entry for convenience
      Fi = np.zeros((N, N, V))

      t_Fi_start = time.time()

      # Initialize for l=1
      for m in range(0, N):
        if m != i:
          wm = weight_disc[m]
          ind_m = val_ind_map[wm]
          Fi[m, 1, ind_m] = 1

      check_vals = np.round(all_possible-wi, n_digit)
      valid_indices = np.logical_and(check_vals >= weight_min_disc, check_vals <= weight_max_disc)
      invalid_indices = ~valid_indices
      mapped_inds = np.array([val_ind_map[val] for val in check_vals[valid_indices]])

      # For 2 <= l <= K-1
      for l in range(2, K):
        Fi[l-1:i, l, :] = F[l-1:i, l, :]

      for l in range(2, K):
        Fi[max(l-1, i+1):N, l, valid_indices] = F[max(l-1, i+1):N, l, valid_indices] - Fi[max(l-1, i+1):N, l-1, mapped_inds]
        Fi[max(l-1, i+1):N, l, invalid_indices] = F[max(l-1, i+1):N, l, invalid_indices]
        
      print('i={}, small_loop={}'.format(i, time.time()-t_Fi_start))

      t_Ei_s = time.time()

      # Compute l_star s.t. error <= eps
      err = 0
      # print('eps={}'.format(eps))
      if eps > 0:
        l_star = N+1
        while err < eps:
          l_star -= 1
          err += E(i, l_star-1)
      else:
        l_star = N

      t_Ei += time.time()-t_Ei_s

      print('l_star = {}'.format(l_star))

      t_bigloop_s = time.time()

      for m in range( max(i+1, K-1), N ):
        wm = weight_disc[m]
        check_vals = np.round(all_possible-wi+wm, n_digit)
        valid_indices = np.logical_and(check_vals >= weight_min_disc, check_vals <= weight_max_disc)
        mapped_inds = np.array([val_ind_map[val] for val in check_vals[valid_indices]])
        H_reshaped = H[K:l_star, m][:, np.newaxis]  # (l_star - K, 1)
        updates = Fi[m, K - 1, mapped_inds] * H_reshaped  # (l_star - K, len(mapped_inds))
        Fi[m, K:l_star, valid_indices] = F[m, K:l_star, valid_indices] - updates.T
        Fi[m, K:l_star, ~valid_indices] = F[m, K:l_star, ~valid_indices]

      print('i={}, bigloop={}'.format(i, time.time()-t_Fi_start))
      t_bigloop += (time.time() - t_bigloop_s)

      Gi = np.zeros(N)

      t_Gi = time.time()

      start_ind, end_ind = 0, -1
      if wi > 0: 
        start_val, end_val = -wi, -interval
        start_ind, end_ind = val_ind_map[round(start_val, n_digit)], val_ind_map[round(end_val, n_digit)]
      elif wi < 0:
        start_val, end_val = 0, -wi-interval
        start_ind, end_ind = val_ind_map[round(start_val, n_digit)], val_ind_map[round(end_val, n_digit)]

      for m in range(N):
        if i != m:
          Gi[1:K] += np.sum(Fi[m, 1:K, start_ind:end_ind+1], axis=1)

      for m in range(N):
        if i != m:
            wm = weight_disc[m]
            start_ind, end_ind = 0, -1
            if wi > 0 and wm < wi: 
              start_val, end_val = -wi, -wm-interval
              start_ind, end_ind = val_ind_map[round(start_val, n_digit)], val_ind_map[round(end_val, n_digit)]
            elif wi < 0 and wm > wi:
              start_val, end_val = -wm, -wi-interval
              start_ind, end_ind = val_ind_map[round(start_val, n_digit)], val_ind_map[round(end_val, n_digit)]

        Gi[K:N] += np.sum(Fi[m, K:N, start_ind:end_ind+1], axis=1)

      Gi_l = Gi[1:]/I[1:]

      t_computeGi += (time.time() - t_Gi)

      sv[i] = np.sum(Gi_l)
      if wi < 0:
        sv[i] += 1 # for l=0
        sv[i] = -sv[i]

  print('Computed Fi; Time: {}, Ei_time={}, BigLoop={}, t_computeGi={}'.format(time.time()-t_Fi, t_Ei, t_bigloop, t_computeGi))

  if debug: 
    print('weight_disc', weight_disc)
    print(sv)

  sv_real = np.zeros(N)
  sv_real[rank] = sv
  sv = sv_real

  if debug: 
    print(2*(y_train_few==y_test)-1)
    print(sv)
    print('Sanity check: sum of SV = {}, U(N)-U(empty)={}'.format(
      np.sum(sv) / N, int(np.sum(weight_disc[:K]) >= 0)-1 ))

  return sv


def fastweighted_knn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, K, eps, dis_metric='cosine', kernel='rbf', debug=True):
  
  N = len(y_train_few)
  sv = np.zeros(N)
  n_test = len(y_val_few)

  for i in tqdm(range(n_test)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += fastweighted_knn_shapley_single(x_train_few, y_train_few, x_test, y_test, K, eps, dis_metric, kernel, debug)

  return sv





# x_test, y_test are single data point
def fastweighted_knn_shapley_old_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric='cosine', kernel='rbf', debug=True):

  N = len(y_train_few)
  sv = np.zeros(N)

  C = max(y_train_few)+1

  # Currently only work for binary classification
  assert C==2

  # Currently only work for K>1
  assert K > 1

  # Compute distance
  distance = compute_dist(x_train_few, x_test, dis_metric)

  # Compute weights
  weight = compute_weights(distance, kernel)

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

  # reorder weight_disc based on rank
  rank = np.arange(N).astype(int)
  weight_disc = weight_disc[rank]

  # maximum possible range of weights  
  weight_max_disc, weight_min_disc, N_possible, all_possible, V = get_range(weight_disc, n_digit, interval, K)
  
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


  # set size+1 for each entry for convenience
  F = np.zeros((N, N+1, V))

  # Initialize for l=1
  for m in range(0, N):
    wm = weight_disc[m]
    ind_m = val_ind_map[wm]
    F[m, 1, ind_m] = 1
      
  # For 2 <= l <= K-1
  for l in range(2, K):
    for m in range(l-1, N):
      for j, s in enumerate(all_possible):
        wm = weight_disc[m]
        check_val = np.round(s-wm, n_digit)
        if check_val < weight_min_disc or check_val > weight_max_disc:
          F[m, l, j] = 0
        else:
          ind_sm = val_ind_map[check_val]
          for t in range(m):
            F[m, l, j] += F[t, l-1, ind_sm]

  # For K <= l <= N
  for l in range(K, N+1):
    for m in range(l-1, N):
      for j, s in enumerate(all_possible):
        for t in range(m):
          F[m, l, j] += F[t, K-1, j] * comb(N-m, l-K)


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
          for j, s in enumerate(all_possible):
            if i < m:
              check_val = np.round(s-wi, n_digit)
              if check_val < weight_min_disc or check_val > weight_max_disc:
                Fi[m, l, j] = F[m, l, j]
              else:
                ind_sm = val_ind_map[check_val]
                Fi[m, l, j] = F[m, l, j] - Fi[m, l-1, ind_sm]
            elif i > m:
              Fi[m, l, j] = F[m, l, j]

      # For K <= l <= N-1
      for l in range(K, N):
        for m in range( max(i+1, l-1), N ):
          for j, s in enumerate(all_possible):
            wm = weight_disc[m]
            check_val = np.round(s-wi+wm, n_digit)
            if check_val < weight_min_disc or check_val > weight_max_disc:
              Fi[m, l, j] = F[m, l, j]
            else:
              ind_sm = val_ind_map[check_val]
              Fi[m, l, j] = F[m, l, j] - Fi[m, K-1, ind_sm] * comb(N-m, l-K)

      Gi = np.zeros(N)

      if wi > 0:
        check_range = np.round(np.linspace(-wi, 0, int(np.round(wi/interval))+1), n_digit)
      elif wi < 0:
        check_range = np.round(np.linspace(0, -wi, int(np.round(-wi/interval))+1), n_digit)
      else:
        check_range = []

      for l in range(1, K):
        for m in range(N):
          if i != m:
            for s in check_range:
              if s not in val_ind_map:
                pass
              else:
                ind = val_ind_map[s]
                Gi[l] += Fi[m, l, ind]

      for l in range(K, N):
        for m in range(N):
          if i != m:
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
    print(2*(y_train_few==y_test)-1)
    print(sv)
    print(np.sum(sv))

  return sv



def fastweighted_knn_shapley_old(x_train_few, y_train_few, x_val_few, y_val_few, K, dis_metric='cosine', kernel='rbf', debug=True):
  
  N = len(y_train_few)
  sv = np.zeros(N)
  n_test = len(y_val_few)

  for i in tqdm(range(n_test)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += fastweighted_knn_shapley_old_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric, kernel, debug)

  return sv







# x_test, y_test are single data point
def approxfastweighted_knn_shapley_single(x_train_few, y_train_few, x_test, y_test, K, eps, dis_metric='cosine', kernel='rbf', debug=True):

  N = len(y_train_few)
  sv = np.zeros(N)

  C = max(y_train_few)+1

  # Currently only work for binary classification
  assert C==2

  # Currently only work for K>1
  assert K > 1

  # Compute distance
  distance = compute_dist(x_train_few, x_test, dis_metric)

  # Compute weights
  weight = compute_weights(distance, kernel)

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

  # reorder weight_disc based on rank
  rank = np.argsort(distance)
  weight_disc = weight_disc[rank]

  # maximum possible range of weights
  weight_max_disc, weight_min_disc, N_possible, all_possible, V = get_range(weight_disc, n_digit, interval, K)

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


  # set size+1 for each entry for convenience
  F = np.zeros((N, N+1, V))

  # Initialize for l=1
  for m in range(0, N):
    wm = weight_disc[m]
    ind_m = val_ind_map[wm]
    F[m, 1, ind_m] = 1
      
  # For 2 <= l <= K-1
  for l in range(2, K):
    for m in range(l-1, N):
      for j, s in enumerate(all_possible):
        wm = weight_disc[m]
        check_val = np.round(s-wm, n_digit)
        if check_val < weight_min_disc or check_val > weight_max_disc:
          F[m, l, j] = 0
        else:
          ind_sm = val_ind_map[check_val]
          for t in range(m):
            F[m, l, j] += F[t, l-1, ind_sm]

  # For K <= l <= N
  for l in range(K, N+1):
    for m in range(l-1, N):
      for j, s in enumerate(all_possible):
        for t in range(m):
          F[m, l, j] += F[t, K-1, j] * comb(N-m, l-K)

  # error bound for setting Gi(l)=0; TODO: improve the efficiency of computing E(i, l)
  def E(i, l):
    i = i+1
    return np.sum( [comb(i-1, j)*comb(N-i, l-j) for j in range(K)] / comb(N-1, l) )

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
          for j, s in enumerate(all_possible):
            if i < m:
              check_val = np.round(s-wi, n_digit)
              if check_val < weight_min_disc or check_val > weight_max_disc:
                Fi[m, l, j] = F[m, l, j]
              else:
                ind_sm = val_ind_map[check_val]
                Fi[m, l, j] = F[m, l, j] - Fi[m, l-1, ind_sm]
            elif i > m:
              Fi[m, l, j] = F[m, l, j]

      # Compute l_star s.t. error <= eps
      err = 0
      l_star = N
      while err < eps:
        l_star -= 1
        err += E(i, l_star)

      print('l_star = {}'.format(l_star))

      # For K <= l <= N-1
      for l in range(K, l_star+1):
        for m in range( max(i+1, l-1), N ):
          for j, s in enumerate(all_possible):
            wm = weight_disc[m]
            check_val = np.round(s-wi+wm, n_digit)
            if check_val < weight_min_disc or check_val > weight_max_disc:
              Fi[m, l, j] = F[m, l, j]
            else:
              ind_sm = val_ind_map[check_val]
              Fi[m, l, j] = F[m, l, j] - Fi[m, K-1, ind_sm] * comb(N-m, l-K)

      Gi = np.zeros(N)

      if wi > 0:
        check_range = np.round(np.linspace(-wi, -interval, int(np.round(wi/interval))), n_digit)
      elif wi < 0:
        check_range = np.round(np.linspace(0, -wi-interval, int(np.round(-wi/interval))), n_digit)
      else:
        check_range = []

      for l in range(1, K):
        for m in range(N):
          if i != m:
            for s in check_range:
              if s not in val_ind_map:
                pass
              else:
                ind = val_ind_map[s]
                Gi[l] += Fi[m, l, ind]

      for l in range(K, l_star+1):
        for m in range(N):
          wm = weight_disc[m]

          if wi > 0 and wm < wi:
            check_range = np.round(np.linspace(-wi, -wm-interval, int(np.round((wi-wm)/interval))), n_digit)
          elif wi < 0 and wm > wi:
            check_range = np.round(np.linspace(-wm, -wi-interval, int(np.round((wm-wi)/interval))), n_digit)
          else:
            check_range = []

          for s in check_range:
            if s not in val_ind_map:
              pass
            else:
              ind = val_ind_map[s]
              Gi[l] += Fi[m, l, ind]

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
    print(2*(y_train_few==y_test)-1)
    print(sv)
    print(np.sum(sv))

  return sv



def approxfastweighted_knn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, K, eps, dis_metric='cosine', kernel='rbf', debug=True):
  
  N = len(y_train_few)
  sv = np.zeros(N)
  n_test = len(y_val_few)

  for i in tqdm(range(n_test)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += approxfastweighted_knn_shapley_single(x_train_few, y_train_few, x_test, y_test, K, eps, dis_metric, kernel, debug)

  return sv



