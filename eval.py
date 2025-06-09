
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
from modelAndPerformances import Transformer, eval_step, data_with_loader, performances_to_pd, make_data, MyDataSet
from hyperparametricSelection import hyperparameterSelection

# Transformer Parameters
d_model = 64
d_ff = 512
n_heads, n_layers, d_k = hyperparameterSelection()

batch_size = 1024
epochs = 50
threshold = 0.5

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dataIndependent = pd.read_csv("data/testData/independent_set.csv")
dataExternal = pd.read_csv("data/testData/external_set.csv")
pep_inputs, hla_inputs, labels = make_data(dataIndependent)
dataset = MyDataSet(pep_inputs, hla_inputs, labels)
loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=(False), num_workers=0)
independent_data, independent_pep_inputs, independent_hla_inputs, independent_labels, independent_loader = dataIndependent, pep_inputs, hla_inputs, labels, loader

pep_inputs, hla_inputs, labels = make_data(dataExternal)
dataset = MyDataSet(pep_inputs, hla_inputs, labels)
loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=(False), num_workers=0)
external_data, external_pep_inputs, external_hla_inputs, external_labels, external_loader = dataExternal, pep_inputs, hla_inputs, labels, loader

criterion = nn.CrossEntropyLoss()

independent_fold_metrics_list, external_fold_metrics_list = [], []
ys_independent_fold_dict, ys_external_fold_dict = {}, {}

print(f'--- Evaluating Best Model ---')
path_saver = f'./model/best_model.pt'

if not os.path.exists(path_saver):
    print(f'Model not found at {path_saver}')
model = Transformer(d_model, d_k, n_layers, n_heads, d_ff).to(device)
model.load_state_dict(torch.load(path_saver, map_location=device))
model_eval = model.eval()

ys_res_independent, loss_res_independent_list, metrics_res_independent = eval_step(
    model_eval, independent_loader, criterion, threshold, use_cuda
)
ys_res_external, loss_res_external_list, metrics_res_external = eval_step(
    model_eval, external_loader, criterion, threshold, use_cuda
)

independent_fold_metrics_list.append(metrics_res_independent)
external_fold_metrics_list.append(metrics_res_external)

print('****Independent set:')
print(performances_to_pd(independent_fold_metrics_list))
print('****External set:')
print(performances_to_pd(external_fold_metrics_list))