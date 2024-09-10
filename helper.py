import torch
import numpy as np 
import pickle
import warnings

from random import randint

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

import config


big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'


def scale_init_param(net, factor=1):

  for param in net.parameters():
    param.data *= factor

  return net



def kmeans_f1score(value_array, cluster=False):

  n_data = len(value_array)

  if cluster:
    X = value_array.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    min_cluster = min(kmeans.cluster_centers_.reshape(-1))
    pred = np.zeros(n_data)
    pred[value_array < min_cluster] = 1
  else:
    threshold = np.sort(value_array)[int(0.1*n_data)]
    pred = np.zeros(n_data)
    pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return f1_score(true, pred)



def kmeans_aucroc(value_array, cluster=False):

  n_data = len(value_array)

  # if cluster:
  #   X = value_array.reshape(-1, 1)
  #   kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
  #   min_cluster = min(kmeans.cluster_centers_.reshape(-1))
  #   pred = np.zeros(n_data)
  #   pred[value_array < min_cluster] = 1
  # else:
  #   threshold = np.sort(value_array)[int(0.1*n_data)]
  #   pred = np.zeros(n_data)
  #   pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return roc_auc_score( true, - value_array )


def kmeans_aupr(value_array, cluster=False):

  n_data = len(value_array)

  if cluster:
    X = value_array.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    min_cluster = min(kmeans.cluster_centers_.reshape(-1))
    pred = np.zeros(n_data)
    pred[value_array < min_cluster] = 1
  else:
    threshold = np.sort(value_array)[int(0.1*n_data)]
    pred = np.zeros(n_data)
    pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return average_precision_score(true, pred)




"""
def kmeans_f1score(value_array):

  n_data = len(value_array)

  X = value_array.reshape(-1, 1)
  kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
  min_cluster = min(kmeans.cluster_centers_.reshape(-1))
  pred = np.zeros(n_data)
  pred[value_array < min_cluster] = 1
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return f1_score(true, pred)
"""


"""
def load_value_args(value_type, args):

  if args.dataset == 'Dog_vs_CatFeature':

    if args.value_type == 'LeastCore':
      save_name = save_dir + 'Banzhaf_GT_Dog_vs_CatFeature_MLP_Ndata2000_Nval2000_Nsample100000_BS128_Nrepeat5_FR0.1.data'
      value_arg = pickle.load( open(save_name, 'rb') )

      for i, x in enumerate(value_arg['X_feature']):
        value_arg['X_feature'][i] = x[x<args.n_data]
    else:
      save_name = save_dir + '{}_Dog_vs_CatFeature_MLP_Ndata2000_Nval2000_Nsample100000_BS128_Nrepeat5_FR0.1.data'.format( value_type )
      value_arg = pickle.load( open(save_name, 'rb') )

    value_arg['y_feature'] = np.mean( np.array(value_arg['y_feature']), axis=1 )

  else:

    save_name = save_dir + '{}_{}_Logistic_Ndata200_Nval2000_Nsample2000_FR0.1_Seed{}.data'.format(value_type, args.dataset, args.random_state)
    value_arg = pickle.load( open(save_name, 'rb') )
    if 'y_feature' in value_arg.keys():
      value_arg['y_feature'] = np.array(value_arg['y_feature'])

  return value_arg
"""


def load_value_args(value_type, args):

  if value_type == 'BetaShapley':
    base_value_type = 'Shapley_Perm'
  else:
    base_value_type = value_type


  if value_type in ['FixedCard_MC', 'FixedCard_MSR', 'FixedCard_MSRPerm']:
    
    save_dir = 'FixedCard-sample/'

    if args.dataset in OpenML_dataset:

      save_name = save_dir + 'FixedCard_MSR_{}_MLP_Ndata200_Nval500_Nsample10000_BS128_LR0.01_Nrepeat5_FR0.1_card5_Seed1.data'.format(args.dataset)

    elif args.dataset in big_dataset:
      
      save_name = save_dir + '{}_{}_{}_Ndata2000_Nval2000_Nsample{}_BS128_LR0.001_Nrepeat5_FR0.1_card{}_Seed1.data'.format(
        value_type, args.dataset, args.model_type, args.n_sample, args.card)

      """
      save_name = save_dir + '{}_{}_{}_Ndata{}_Nval{}_Nsample{}_BS{}_LR{}_Nrepeat{}_FR{}_card{}_Seed{}.data'.format(
        value_type, args.dataset, args.model_type, args.n_data, args.n_val, args.n_sample, args.batch_size, args.lr, 5, args.flip_ratio, args.card, args.random_state )
      """

    value_arg = pickle.load( open(save_name, 'rb') )

    seed = args.random_state

    """
    while 1:
      seed += 1
      save_name = save_dir + '{}_{}_{}_Ndata{}_Nval{}_Nsample{}_BS{}_LR{}_Nrepeat{}_FR{}_card{}_Seed{}.data'.format(
        value_type, args.dataset, args.model_type, args.n_data, args.n_val, args.n_sample, args.batch_size, args.lr, 5, args.flip_ratio, args.card, seed )
      if exists(save_name) and value_type == 'FixedCard_MC':
        value_arg_add = pickle.load( open(save_name, 'rb') )
        for i in range(args.n_data):
          value_arg[i] = value_arg[i] + value_arg_add[i]
      else:
        break
    """

  elif value_type in ['FZ20']:

    dir_rebuttal = '/home/tw8948/scaling-law/result_rebuttal/'

    if args.dataset in OpenML_dataset:
      save_name = dir_rebuttal + 'FZ20_{}_MLP_Ndata200_Nval200_Nsample10000_BS128_LR0.01_Nrepeat5_FR0.1_card10_Seed1.data'.format(args.dataset)
    elif args.dataset in ['MNIST', 'FMNIST']:
      save_name = dir_rebuttal + 'FZ20_{}0.5_SmallCNN_Ndata2000_Nval2000_Nsample10000_BS128_LR0.001_Nrepeat5_FR0.1_card10_Seed1.data'.format(args.dataset)
    elif args.dataset in ['Dog_vs_CatFeature']:
      save_name = dir_rebuttal + 'FZ20_{}_MLP_Ndata2000_Nval2000_Nsample10000_BS128_LR0.001_Nrepeat5_FR0.1_card10_Seed1.data'.format(args.dataset)

    value_arg = pickle.load( open(save_name, 'rb') )
    value_arg['n_data'] = args.n_data

  else:
    
    save_dir = 'baseline_result/'

    if args.dataset in ['Dog_vs_CatFeature']:
      save_name = save_dir + '{}_Dog_vs_CatFeature_MLP_Ndata2000_Nval2000_Nsample10000_BS128_Nrepeat5_FR0.1_Seed1.data'.format(base_value_type)
    
    elif args.dataset in ['MNIST', 'FMNIST']:
      save_name = save_dir + '{}_{}_SmallCNN_Ndata2000_Nval2000_Nsample40000_BS128_LR0.001_Nrepeat5_FR0.1.data'.format(base_value_type, args.dataset)

    elif args.dataset in big_dataset:
      save_name = save_dir+'{}_{}_{}_Ndata{}_Nval{}_Nsample{}_BS{}_LR{}_Nrepeat{}_FR{}.data'.format(
            base_value_type, args.dataset, args.model_type, args.n_data, args.n_val, args.n_sample, args.batch_size, args.lr, 5, args.flip_ratio)
    elif args.dataset in OpenML_dataset:
      if args.noisydata:
        save_name = save_dir+'{}_{}_MLP_Ndata2000_Nval200_Nsample10000_BS32_LR0.01_Nrepeat1_FR0.1_Seed42_NoisyDataTrue.data'.format(base_value_type, args.dataset)
      else:
        save_name = save_dir+'{}_{}_MLP_Ndata2000_Nval200_Nsample10000_BS32_LR0.01_Nrepeat1_FR0.1_Seed42.data'.format(base_value_type, args.dataset)

    else:
      save_name = save_dir+'{}_{}_{}_Ndata{}_Nval{}_Nsample{}_FR{}.data'.format(
            base_value_type, args.dataset, args.model_type, args.n_data, args.n_val, args.n_sample, args.flip_ratio)

    value_arg = pickle.load( open(save_name, 'rb') )

  if 'y_feature' in value_arg.keys():
    value_arg['y_feature'] = np.array(value_arg['y_feature'])

  if value_type == 'BetaShapley':
    value_arg['alpha'] = args.alpha
    value_arg['beta'] = args.beta

  return value_arg


# args: a dictionary
def compute_value(value_type, args):
  if value_type == 'Shapley_Perm':
    sv = shapley_permsampling_from_data(args['X_feature'], args['y_feature'], args['n_data'], v0=args['sv_baseline'])
  elif value_type == 'BetaShapley':
    sv = betasv_permsampling_from_data(args['X_feature'], args['y_feature'], args['n_data'], args['alpha'], args['beta'], v0=args['sv_baseline'])
  elif value_type == 'Banzhaf_GT':
    sv = banzhaf_grouptest_bias_from_data(args['X_feature'], args['y_feature'], args['n_data'], dummy=True)
  elif value_type == 'LOO':
    sv = compute_loo(args['y_feature'], args['u_total'])
  elif value_type == 'KNN':
    sv = args['knn']
  elif value_type == 'Uniform':
    sv = np.ones(args['n_data'])
  elif value_type == 'Shapley_GT':
    sv = shapley_grouptest_from_data(args['X_feature'], args['y_feature'], args['n_data'])
  elif value_type == 'LeastCore':
    sv = banzhaf_grouptest_bias_from_data(args['X_feature'], args['y_feature'], args['n_data'], dummy=False)
  elif value_type == 'FixedCard_MC':
    sv = compute_FixedCard_MC_from_data(args, args['n_data'], args['func'])
  elif value_type in ['FixedCard_MSR', 'FixedCard_MSRPerm']:
    sv = compute_FixedCard_MSR_from_data(args, args['n_data'], args['func'])
  elif value_type == 'FZ20':
    sv = banzhaf_grouptest_bias_from_data(args['X_feature'], args['y_feature'], args['n_data'], dummy=False)

  return sv



def normalize(val):
  v_max, v_min = np.max(val), np.min(val)
  val = (val-v_min) / (v_max - v_min)
  return val


def shapley_permsampling_from_data(X_feature, y_feature, n_data, v0=0.1):

  n_sample = len(y_feature)
  n_perm = int( n_sample // n_data )

  if n_sample%n_data > 0: 
    print('WARNING: n_sample cannot be divided by n_data')

  sv_vector = np.zeros(n_data)
  
  for i in range(n_perm):
    for j in range(0, n_data):
      target_ind = X_feature[i*n_data+j][-1]
      if j==0:
        without_score = v0
      else:
        without_score = y_feature[i*n_data+j-1]
      with_score = y_feature[i*n_data+j]
      
      sv_vector[target_ind] += (with_score-without_score)
  
  return sv_vector / n_perm


def beta_constant(a, b):
    '''
    the second argument (b; beta) should be integer in this function
    '''
    beta_fct_value=1/a
    for i in range(1,b):
        beta_fct_value=beta_fct_value*(i/(a+i))
    return beta_fct_value


def compute_weight_list(m, alpha=1, beta=1):
    '''
    Given a prior distribution (beta distribution (alpha,beta))
    beta_constant(j+1, m-j) = j! (m-j-1)! / (m-1)! / m # which is exactly the Shapley weights.
    # weight_list[n] is a weight when baseline model uses 'n' samples (w^{(n)}(j)*binom{n-1}{j} in the paper).
    '''
    weight_list=np.zeros(m)
    normalizing_constant=1/beta_constant(alpha, beta)
    for j in np.arange(m):
        # when the cardinality of random sets is j
        weight_list[j]=beta_constant(j+alpha, m-j+beta-1)/beta_constant(j+1, m-j)
        weight_list[j]=normalizing_constant*weight_list[j] # we need this '/m' but omit for stability # normalizing
    return weight_list


def betasv_permsampling_from_data(X_feature, y_feature, n_data, a, b, v0=0.1):

  n_sample = len(y_feature)
  n_perm = int( n_sample // n_data )

  if n_sample%n_data > 0: 
    print('WARNING: n_sample cannot be divided by n_data')

  """
  weight_vector = np.zeros(n_data)
  for j in range(1, n_data+1):
    w = n_data * beta(j+b-1, n_data-j+a) / beta(a, b) * comb(n_data-1, j-1)
    weight_vector[j-1] = w
  """
  weight_vector = compute_weight_list(n_data, alpha=a, beta=b)
  #print(weight_vector[:1000])

  sv_vector = np.zeros(n_data)
  
  for i in range(n_perm):
    for j in range(0, n_data):
      target_ind = X_feature[i*n_data+j][-1]
      if j==0:
        without_score = v0
      else:
        without_score = y_feature[i*n_data+j-1]
      with_score = y_feature[i*n_data+j]
      
      sv_vector[target_ind] += weight_vector[j]*(with_score-without_score)

  return sv_vector / n_perm



def banzhaf_grouptest_bias_from_data(X_feature, y_feature, n_data, dummy=True):

  n_sample = len(y_feature)

  if dummy:
    N = n_data+1
  else:
    N = n_data

  A = np.zeros((n_sample, N))
  B = y_feature

  for t in range(n_sample):
    A[t][X_feature[t]] = 1

  sv_approx = np.zeros(n_data)

  for i in range(n_data):
    if np.sum(A[:, i]) == n_sample:
      sv_approx[i] = np.dot( A[:, i], B ) / n_sample
    elif np.sum(A[:, i]) == 0:
      sv_approx[i] = - np.dot( (1-A[:, i]), B ) / n_sample
    else:
      sv_approx[i] = np.dot(A[:, i], B)/np.sum(A[:, i]) - np.dot(1-A[:, i], B)/np.sum(1-A[:, i])

  return sv_approx


def sample_utility_multiple(x_train, y_train, x_test, y_test, utility_func, n_repeat):

  acc_lst = []

  for _ in range(n_repeat):
    acc = utility_func(x_train, y_train, x_test, y_test)
    acc_lst.append(acc)

  return acc_lst


def sample_utility_shapley_perm(n_perm, utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  X_feature_test = []
  y_feature_test = []
  
  for k in range(n_perm):

    print('Permutation {} / {}'.format(k, n_perm))
    perm = np.random.permutation(range(n_data))

    for i in range(1, n_data+1):
      subset_index = perm[:i]
      X_feature_test.append(subset_index)
      y_feature_test.append(utility_func(x_train[subset_index], y_train[subset_index], x_val, y_val))

  return X_feature_test, y_feature_test


def sample_L_utility_shapley_perm(n_perm, du_model, n_data):

  X_feature_test = []
  y_feature_test = []
  
  for k in range(n_perm):

    print('Permutation {} / {}'.format(k, n_perm))
    perm = np.random.permutation(range(n_data))

    for i in range(1, n_data+1):
      subset_index = perm[:i]
      X_feature_test.append(subset_index)

      subset_bin = np.zeros((1, 200))
      subset_bin[0, subset_index] = 1

      y = du_model(torch.tensor(subset_bin).float().cuda()).cpu().detach().numpy().reshape(-1)
      y_feature_test.append(y[0])

  return X_feature_test, np.array(y_feature_test)




def uniformly_subset_sample(dataset):

  sampled_set = []

  for data in dataset:
    if randint(0, 1) == 1:
      sampled_set.append(data)

  return sampled_set


def sample_utility_banzhaf_mc(n_sample, utility_func, utility_func_args, target_ind):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  n_sample_per_data = int( n_sample / 2 )

  # utility set will store tuples (with/without target index)
  utility_set = []

  dataset = np.arange(n_data)
  leave_one_out_set = np.delete(dataset, target_ind)

  for _ in range(n_sample_per_data):
    sampled_idx_without = np.array(uniformly_subset_sample(leave_one_out_set))
    utility_without = utility_func(x_train[sampled_idx_without], y_train[sampled_idx_without], x_val, y_val)
    sampled_idx_with = np.array( list(sampled_idx_without) + [target_ind] )
    utility_with = utility_func(x_train[sampled_idx_with], y_train[sampled_idx_with], x_val, y_val)

    to_be_store = { 'ind': sampled_idx_without, 'u_without': utility_without, 'u_with': utility_with }

    utility_set.append(to_be_store)

  return utility_set



def sample_utility_FixedCard_MC(n_sample, utility_func, utility_func_args, target_ind, size):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  n_sample_per_data = int( n_sample / 2 )

  # utility set will store tuples (with/without target index)
  utility_set = []

  dataset = np.arange(n_data)
  leave_one_out_set = np.delete(dataset, target_ind)

  for _ in range(n_sample_per_data):
    
    sampled_idx_without = np.array(uniformly_subset_givensize(leave_one_out_set, size))

    utility_without = utility_func(x_train[sampled_idx_without], y_train[sampled_idx_without], x_val, y_val)
    sampled_idx_with = np.array( list(sampled_idx_without) + [target_ind] )
    utility_with = utility_func(x_train[sampled_idx_with], y_train[sampled_idx_with], x_val, y_val)

    to_be_store = { 'ind': sampled_idx_without, 'u_without': utility_without, 'u_with': utility_with }

    utility_set.append(to_be_store)

  return utility_set



def sample_utility_FixedCard_MSR(n_sample, utility_func, utility_func_args, card):

  x_train, y_train, x_val, y_val = utility_func_args

  to_be_store = {}

  n_data = len(y_train)
  dataset = np.arange(n_data)

  X_feature_small = []
  y_feature_small = []

  n_sample_per_card = int( n_sample / 2 )

  for t in range(n_sample_per_card):
    subset_ind = np.array(uniformly_subset_givensize(dataset, card))
    X_feature_small.append( subset_ind )
    y_feature_small.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )

  to_be_store['small'] = [X_feature_small, y_feature_small]


  X_feature_large = []
  y_feature_large = []

  for t in range(n_sample_per_card):
    subset_ind = np.array(uniformly_subset_givensize(dataset, card+1))
    X_feature_large.append( subset_ind )
    y_feature_large.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )

  to_be_store['large'] = [X_feature_large, y_feature_large]
  
  return to_be_store


def sample_utility_FixedCard_MSR_permutation(n_sample, utility_func, utility_func_args, card):

  x_train, y_train, x_val, y_val = utility_func_args

  to_be_store = {}

  n_data = len(y_train)
  k = card
  n = n_data
  
  n_perm_large = int( np.ceil( (n-k)*(k+1) / (n*(n+1)) * n_sample ) )
  n_perm_small = int( np.ceil( k*(k+1) / (n*(n+1)) * n_sample ) )
  
  print('n_perm_small={}, n_perm_large={}'.format(n_perm_small, n_perm_large))

  X_feature_large = []
  y_feature_large = []
  
  for t in range(n_perm_large):
    
    random_perm = np.random.permutation(n)
    index = k+1
    
    while index < n:
      subset_ind = random_perm[index-(k+1):index]
      X_feature_large.append( subset_ind )
      y_feature_large.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
      index += (k+1)
      
    if index > n-1:
      subset_ind = random_perm[n-(k+1):n]
      X_feature_large.append( subset_ind )
      y_feature_large.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
        
  to_be_store['large'] = [X_feature_large, y_feature_large]
  
          
  X_feature_small = []
  y_feature_small = []
  
  for t in range(n_perm_small):
    
    random_perm = np.random.permutation(n)
    index = k
    
    while index < n:
      subset_ind = random_perm[index-(k):index]
      X_feature_small.append( subset_ind )
      y_feature_small.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
      index += (k)
      
    if index > n-1:
      subset_ind = random_perm[n-(k):n]
      X_feature_small.append( subset_ind )
      y_feature_small.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
      
  to_be_store['small'] = [X_feature_small, y_feature_small]
  
  return to_be_store



def compute_FixedCard_MC_from_data(ret, n_data, func):

  n_sample_per_data = len(ret[0])
  # print('total used sample = {}'.format(n_sample_per_data*2*n_data))

  bz_vector = np.zeros(n_data)

  for i in range(n_data):
    ret_i = ret[i]
    for j in range(n_sample_per_data):
      bz_vector[i] += ( (func(ret_i[j]['u_with'])-func(ret_i[j]['u_without'])) / n_sample_per_data )

  return bz_vector



def compute_FixedCard_MSR_from_data(args, n_data, func):

  n_sample, _ = np.array(args['small'][0]).shape

  A = np.zeros( (n_sample, n_data) )
  B = np.array( [ func(x) for x in args['small'][1] ] )

  small_ind = np.array(args['small'][0])

  for t in range(n_sample):
    A[t][small_ind[t]] = 1

  avg_small = np.zeros(n_data)

  for i in range(n_data):
    if np.sum(A[:, i]) == n_sample:
      avg_small[i] = 0
      warnings.warn("WARNING: Sample Might Be Insufficient")
    else:
      avg_small[i] = np.dot(1-A[:, i], B) / np.sum(1-A[:, i])

  n_sample, _ = np.array(args['large'][0]).shape
  
  A = np.zeros( (n_sample, n_data) )
  B = np.array( [ func(x) for x in args['large'][1] ] )

  large_ind = np.array(args['large'][0])

  for t in range(n_sample):
    A[t][large_ind[t]] = 1

  avg_large = np.zeros(n_data)

  for i in range(n_data):
    if np.sum(A[:, i]) == 0:
      avg_large[i] = 0
      warnings.warn("WARNING: Sample Might Be Insufficient")
    else:
      avg_large[i] = np.dot(A[:, i], B) / np.sum(A[:, i])

  sv_approx = avg_large - avg_small

  return sv_approx




# Implement Dummy Data Point Idea
def sample_utility_banzhaf_gt(n_sample, utility_func, utility_func_args, dummy=False):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  if dummy:
    N = n_data + 1
  else:
    N = n_data

  X_feature_test = []
  y_feature_test = []


  for t in range(n_sample):

    # Uniformly sample data points from N data points
    subset_ind = np.array(uniformly_subset_sample( np.arange(N) )).astype(int)

    X_feature_test.append(subset_ind)

    if dummy:
      subset_ind = subset_ind[subset_ind < n_data]

    y_feature_test.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
  
  return X_feature_test, y_feature_test


def sample_utility_banzhaf_gt_new(n_sample, utility_func, utility_func_args, dummy=False):

  trainset, testset = utility_func_args
  n_data = len(trainset)

  if dummy:
    N = n_data + 1
  else:
    N = n_data

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):

    # Uniformly sample data points from N data points
    subset_ind = np.array(uniformly_subset_sample( np.arange(N) )).astype(int)
    X_feature_test.append(subset_ind)

    if dummy: subset_ind = subset_ind[subset_ind < n_data]

    y_feature_test.append( utility_func(trainset, testset, subset_ind) )
  
  return X_feature_test, y_feature_test


# Implement Dummy Data Point Idea
def sample_utility_shapley_gt_new(n_sample, utility_func, utility_func_args):

  trainset, testset = utility_func_args
  n_data = len(trainset)

  N = n_data + 1
  Z = np.sum([1/k+1/(N-k) for k in range(1, N)])
  q = [1/Z * (1/k+1/(N-k)) for k in range(1, N)]

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):
    # Randomly sample size from 1,...,N-1
    size = np.random.choice(np.arange(1, N), p=q)

    # Uniformly sample k data points from N data points
    subset_ind = np.random.choice(np.arange(N), size, replace=False)

    X_feature_test.append(subset_ind)

    subset_ind = subset_ind[subset_ind < n_data]

    if size == 0:
      y_feature_test.append( [0.1]*len(y_feature_test[0]) )
    else:
      y_feature_test.append( utility_func(trainset, testset, subset_ind) )
  
  return X_feature_test, y_feature_test




def sample_utility_FZ20(n_sample, utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  N = len(y_train)
  card = int(N*0.7)

  X_feature_test = []
  y_feature_test = []

  for _ in range(n_sample):
    subset_ind = np.array(uniformly_subset_givensize(np.arange(N), card)).astype(int)
    X_feature_test.append( subset_ind )
    y_feature_test.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
  
  return X_feature_test, y_feature_test





# Leave-one-out
def sample_utility_loo(utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  N = n_data

  X_feature_test = []
  y_feature_test = []

  u_total = utility_func(x_train, y_train, x_val, y_val)

  for i in range(N):

    loo_index = np.ones(N)
    loo_index[i] = 0
    loo_index = loo_index.nonzero()[0]

    X_feature_test.append( loo_index )
    y_feature_test.append( utility_func(x_train[loo_index], y_train[loo_index], x_val, y_val) )

  return X_feature_test, y_feature_test, u_total


# y_feature is 1-dim array, u_total is scalar
def compute_loo(y_feature, u_total):
  score = np.zeros(len(y_feature))
  for i in range(len(y_feature)):
    score[i] = u_total - y_feature[i]
  return score




# For function that is monotonically decrease
def binarySearch(fun, lo, up, tol=1e-10):
    x = lo
    count = 0
    while np.abs(fun(x)) > tol:
        count += 1
        if fun(x)<0:
            up = x
            x = (up+lo)/2
        else:
            lo = x
            x = (lo+up)/2
    print('binarySearch Count:', count)
    return x



# # x_test, y_test are single data point
# def private_knn_banzhaf_single_sigmalst(x_train_few, y_train_few, x_test, y_test, tau=0, K0=10, sigma_lst=0, q=1):

#   N = len(y_train_few)
#   sv = np.zeros(N)
#   C = max(y_train_few)+1

#   sub_ind_bool = (np.random.choice([0, 1], size=N, p=[1-q, q])).astype(bool)
#   sub_ind = np.where(sub_ind_bool)[0]

#   distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])
#   rank_all = np.argsort(distance)

#   if tau == 0:
#     tau = x_train_few[rank_all[K0-1]]
#   Itau_all = (distance <= tau).nonzero()[0]

#   # Itau_subset: index in terms of subset
#   distance_subset = distance[sub_ind]
#   Itau_subset = (distance_subset <= tau).nonzero()[0]

#   Ct = len(Itau_subset)
#   Ca = np.sum( y_train_few[sub_ind[Itau_subset]] == y_test )
#   Ct, Ca = np.round(Ct), np.round(Ca)
#   Ct, Ca = max(Ct, 0), max(Ca, 0)

#   for i in range(N):

#     sigma = sigma_lst[i]

#     Ct_i = Ct + np.random.normal(scale=sigma)
#     Ca_i = Ca + np.random.normal(scale=sigma)

#     if i in Itau_all:

#       if i in sub_ind:
#         Ct_i = max(1, Ct_i)
#         Ca_i = Ca_i
#         if y_test==y_train_few[i]:
#           Ca_i = max(1, Ca_i)
#       else:
#         Ct_i = Ct_i + 1
#         Ct_i = max(1, Ct_i)
#         Ca_i = Ca_i + int(y_test==y_train_few[i])

#       sv[i] = int(y_test==y_train_few[i]) * (2-2**(1-Ct_i))/Ct_i - 2**(1-Ct_i)/C
#       if Ct_i >= 2:
#         sv[i] += - (Ca_i-int(y_test==y_train_few[i])) * (2-(Ct_i+1)*2**(1-Ct_i)) / (Ct_i*(Ct_i-1))

#   return sv


# from prv_accountant.dpsgd import find_noise_multiplier


# def private_knn_banzhaf_fixeps(x_train_few, y_train_few, x_val_few, y_val_few, tau=0, K0=10, eps=0, q=1, delta=1e-5, q_test=0.1):
  
#   N = len(y_train_few)
#   sv = np.zeros(N)

#   n_test = len(y_val_few)
#   n_test_sub = int(n_test*q_test)
#   test_ind = np.random.choice(range(n_test), size=n_test_sub, replace=False)
#   x_val_few, y_val_few = x_val_few[test_ind], y_val_few[test_ind]

#   n_iter_lst = np.zeros(N)

#   for i in range(n_test_sub):
#     x_test, y_test = x_val_few[i], y_val_few[i]
#     distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])
#     close_lst = (distance<=tau).astype(int)
#     n_iter_lst += close_lst

#   n_iter_lst = n_iter_lst.astype(int)
#   sigma_lst = np.zeros(N)

#   for j, n_compose in enumerate( tqdm(n_iter_lst) ):

#     print(n_compose)

#     sigma = find_noise_multiplier(sampling_probability=q, num_steps=n_compose, target_epsilon=eps, target_delta=delta, eps_error=0.1, mu_max=5000.0)

#     """
#     def get_eps(sigma):
#       mech = PrivateKNN_mech(q, sigma, n_compose)
#       eps0 = mech.get_approxDP(delta=delta)
#       return eps0 - eps
#     sigma = binarySearch(get_eps, 1e-10, 100, tol=1e-5)
#     """

#     sigma_lst[j] = sigma

#   n_iter_lst = np.zeros(N)

#   for i in range(n_test_sub):
#     x_test, y_test = x_val_few[i], y_val_few[i]
#     sv_individual, close_lst = private_knn_banzhaf_single_sigmalst(x_train_few, y_train_few, x_test, y_test, tau, K0, sigma_lst, q)
#     sv += sv_individual
#     n_iter_lst += close_lst

#   n_compose = n_iter_lst[0]
#   mech = PrivateKNN_mech(q, sigma, n_compose)
#   eps = mech.get_approxDP(delta=delta)

#   return sv, eps








def uniformly_subset_givensize(dataset, size):

  sampled_set = np.random.permutation(dataset)

  return sampled_set[:int(size)]


def sample_utility_givensize(n_sample_lst, utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  X_feature_test = []
  y_feature_test = []

  for size in n_sample_lst:

    subset_ind = np.array(uniformly_subset_givensize( np.arange(n_data), size )).astype(int)

    X_feature_test.append(subset_ind)

    y_feature_test.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
  
  return X_feature_test, y_feature_test


# Implement Dummy Data Point Idea
def sample_utility_shapley_gt(n_sample, utility_func, utility_func_args):

  x_train, y_train, x_val, y_val = utility_func_args
  n_data = len(y_train)

  N = n_data + 1
  Z = np.sum([1/k+1/(N-k) for k in range(1, N)])
  q = [1/Z * (1/k+1/(N-k)) for k in range(1, N)]

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):
    # Randomly sample size from 1,...,N-1
    size = np.random.choice(np.arange(1, N), p=q)

    # Uniformly sample k data points from N data points
    subset_ind = np.random.choice(np.arange(N), size, replace=False)

    X_feature_test.append(subset_ind)

    subset_ind = subset_ind[subset_ind < n_data]

    if size == 0:
      y_feature_test.append( [0.1] )
    else:
      y_feature_test.append( utility_func(x_train[subset_ind], y_train[subset_ind], x_val, y_val) )
  
  return X_feature_test, y_feature_test


# Implement Dummy Data Point Idea
def sample_L_utility_shapley_gt(n_sample, du_model, n_data):

  N = n_data + 1
  Z = np.sum([1/k+1/(N-k) for k in range(1, N)])
  q = [1/Z * (1/k+1/(N-k)) for k in range(1, N)]

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):
    # Randomly sample size from 1,...,N-1
    size = np.random.choice(np.arange(1, N), p=q)

    # Uniformly sample k data points from N data points
    subset_ind = np.random.choice(np.arange(N), size, replace=False)

    X_feature_test.append(subset_ind)

    subset_ind = subset_ind[subset_ind < n_data]
    subset_bin = np.zeros((1, n_data))
    subset_bin[0, subset_ind] = 1

    y = du_model(torch.tensor(subset_bin).float().cuda()).cpu().detach().numpy().reshape(-1)
    y_feature_test.append(y[0])

  return X_feature_test, np.array(y_feature_test)


# Implement Dummy Data Point Idea
def sample_L_utility_banzhaf_gt(n_sample, du_model, n_data, dummy=False):

  if dummy:
    N = n_data + 1
  else:
    N = n_data

  X_feature_test = []
  y_feature_test = []

  for t in range(n_sample):

    # Uniformly sample data points from N data points
    subset_ind = np.array(uniformly_subset_sample( np.arange(N) )).astype(int)

    X_feature_test.append(subset_ind)

    if dummy:
      subset_ind = subset_ind[subset_ind < n_data]

    subset_bin = np.zeros((1, n_data))
    subset_bin[0, subset_ind] = 1

    y = du_model(torch.tensor(subset_bin).float().cuda()).cpu().detach().numpy().reshape(-1)
    y_feature_test.append(y[0])

  return X_feature_test, np.array(y_feature_test)



def shapley_grouptest_from_data(X_feature, y_feature, n_data):

  n_sample = len(y_feature)
  N = n_data+1
  Z = np.sum([1/k+1/(N-k) for k in range(1, N)])

  A = np.zeros((n_sample, N))
  B = y_feature

  for t in range(n_sample):
    A[t][X_feature[t]] = 1

  C = {}
  for i in range(N):
    for j in [n_data]:
      C[(i,j)] = Z*(B.dot(A[:,i] - A[:,j]))/n_sample

  sv_last = 0
  sv_approx = np.zeros(n_data)

  for i in range(n_data): 
    sv_approx[i] = C[(i, N-1)] + sv_last
  
  return sv_approx









