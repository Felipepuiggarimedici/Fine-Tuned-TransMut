
import math
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
random.seed(1234)

from numpy import interp
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from modelAndPerformances import Transformer, eval_step, data_with_loader, performances_to_pd

# Transformer Parameters
d_model = 64  # Embedding Size
d_ff = 512 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 1  # number of Encoder of Decoder Layer

batch_size = 1024
epochs = 50
threshold = 0.5

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

independent_data, independent_pep_inputs, independent_hla_inputs, independent_labels, independent_loader = data_with_loader(type_ = 'independent',fold = None,  batch_size = batch_size)
external_data, external_pep_inputs, external_hla_inputs, external_labels, external_loader = data_with_loader(type_ = 'external',fold = None,  batch_size = batch_size)

n_layers = 1
epochs = 50
ep_best = 50
criterion = nn.CrossEntropyLoss()
files = os.listdir("data/")
folds = sum(1 for f in files if os.path.isfile(os.path.join("data/", f)) and 'train' in f and 'fold' in f)

for n_heads in [1, 2, 3, 4, 5]:
    independent_fold_metrics_list, external_fold_metrics_list = [], []
    ys_independent_fold_dict, ys_external_fold_dict = {}, {}

    for fold in folds:
        print(f'--- Evaluating fold {fold} with {n_heads} heads ---')
        path_saver = f'./model/pHLAIformer/model_layer{n_layers}_multihead{n_heads}_fold{fold}.pkl'

        if not os.path.exists(path_saver):
            print(f'Model not found at {path_saver}')
            continue

        model = Transformer(d_model, d_k, n_layers, n_heads, use_cuda, d_ff).to(device)
        model.load_state_dict(torch.load(path_saver))
        model_eval = model.eval()

        ys_res_independent, loss_res_independent_list, metrics_res_independent = eval_step(
            model_eval, independent_loader, fold, ep_best, epochs, criterion, threshold, use_cuda
        )
        ys_res_external, loss_res_external_list, metrics_res_external = eval_step(
            model_eval, external_loader, fold, ep_best, epochs, criterion, threshold, use_cuda
        )

        independent_fold_metrics_list.append(metrics_res_independent)
        external_fold_metrics_list.append(metrics_res_external)

        ys_independent_fold_dict[fold] = ys_res_independent
        ys_external_fold_dict[fold] = ys_res_external

print('****Independent set:')
print(performances_to_pd(independent_fold_metrics_list))
print('****External set:')
print(performances_to_pd(external_fold_metrics_list))