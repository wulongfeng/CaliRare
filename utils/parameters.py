
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Helper functions and utilities for deep learning models
# ---------------------------------------------------------


from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import torch
from torch.autograd import Variable 
import torch.nn.functional as nnf
from torch.utils.data import random_split
from torch.optim import SGD 
from torch.distributions import constraints
import torchvision as torchv
import torchvision.transforms as torchvt
from torch import nn
from torch_geometric.nn import GCNConv
import torchvision.transforms as transforms
from torch.autograd import grad
import scipy.stats as st

#from influence.influence_utils import *

from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import time

torch.manual_seed(1) 


ACTIVATION_DICT = {"ReLU": torch.nn.ReLU(), "Hardtanh": torch.nn.Hardtanh(),
                   "ReLU6": torch.nn.ReLU6(), "Sigmoid": torch.nn.Sigmoid(),
                   "Tanh": torch.nn.Tanh(), "ELU": torch.nn.ELU(),
                   "CELU": torch.nn.CELU(), "SELU": torch.nn.SELU(), 
                   "GLU": torch.nn.GLU(), "LeakyReLU": torch.nn.LeakyReLU(),
                   "LogSigmoid": torch.nn.LogSigmoid(), "Softplus": torch.nn.Softplus()}


def build_architecture(base_model):

    modules          = []

    if base_model.dropout_active:

        modules.append(torch.nn.Dropout(p=base_model.dropout_prob))

    modules.append(torch.nn.Linear(base_model.n_dim, base_model.num_hidden))
    modules.append(ACTIVATION_DICT[base_model.activation])

    for u in range(base_model.num_layers - 1):

        if base_model.dropout_active:

            modules.append(torch.nn.Dropout(p=base_model.dropout_prob))

        modules.append(torch.nn.Linear(base_model.num_hidden, base_model.num_hidden))
        modules.append(ACTIVATION_DICT[base_model.activation])

    modules.append(torch.nn.Linear(base_model.num_hidden, base_model.output_size))

    _architecture    = nn.Sequential(*modules)

    return _architecture


    def __init__(self, hidden_channels, data):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)
        return x


def build_gcn(base_model):

    modules          = []

    if base_model.dropout_active:

        modules.append(torch.nn.Dropout(p=base_model.dropout_prob))

    modules.append(GCNConv(base_model.num_features, base_model.hidden_channels))
    modules.append(ACTIVATION_DICT["ReLU"])

    modules.append(torch.nn.Dropout(p=0.5))
    modules.append(GCNConv(base_model.hidden_channels, 2))

    for u in range(base_model.num_layers - 1):

        if base_model.dropout_active:

            modules.append(torch.nn.Dropout(p=base_model.dropout_prob))

        modules.append(torch.nn.Linear(base_model.num_hidden, base_model.num_hidden))
        modules.append(ACTIVATION_DICT[base_model.activation])

    modules.append(torch.nn.Linear(base_model.num_hidden, base_model.output_size))

    _architecture    = nn.Sequential(*modules)

    return _architecture


def get_number_parameters(model):

    params_ = []

    for param in model.parameters():
    
        params_.append(param)
    
    return stack_torch_tensors(params_).shape[0]    