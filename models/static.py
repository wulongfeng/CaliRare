from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

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
import torchvision.transforms as transforms
from torch.autograd import grad
import scipy.stats as st
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN


from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import time

from utils.parameters import *

torch.manual_seed(1)

from influence.influence_computation import *
from influence.influence_utils import *


class linearRegression(torch.nn.Module):

    def __init__(self, inputSize, outputSize):

        super(linearRegression, self).__init__()

        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):

        out         = self.linear(x)

        return out


class LinearRegression(torch.nn.Module):

    def __init__(self, inputDim=1, outputDim=1, learningRate=0.01, epochs=1000):

        super(LinearRegression, self).__init__()

        self.model         = linearRegression(inputSize=inputDim, outputSize=outputDim)

        if torch.cuda.is_available():

            self.model.cuda()

        self.inputDim      = inputDim   # takes variable 'x'
        self.outputDim     = outputDim  # takes variable 'y'
        self.learningRate  = learningRate
        self.epochs        = epochs
        self.loss_fn       = torch.nn.MSELoss()
        self.optimizer     = torch.optim.SGD(self.model.parameters(), lr=self.learningRate)

    def forward(self, x):

        out         = self.model(x)

        return out

    def fit(self, x_train, y_train, verbosity=True):

        self.X         = torch.tensor(x_train.reshape((-1, self.inputDim))).float()
        self.y         = torch.tensor(y_train).float()
        self.losses    = []

        for epoch in range(self.epochs):

            # Converting inputs and labels to Variable

            if torch.cuda.is_available():

                inputs = Variable(torch.from_numpy(x_train).cuda()).float()
                labels = Variable(torch.from_numpy(y_train).cuda()).float()

            else:

                inputs = Variable(torch.from_numpy(x_train)).float()
                labels = Variable(torch.from_numpy(y_train)).float()


            self.model.zero_grad()

            self.optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = self.model(inputs)

            # get loss for the predicted output
            self.loss = self.loss_fn(outputs, labels)

            # get gradients w.r.t to parameters
            self.loss.backward(retain_graph=True)
            self.losses.append(self.loss.detach().numpy())

            # update parameters
            self.optimizer.step()

            if verbosity:

                print('epoch {}, loss {}'.format(epoch, self.loss.item()))


    def predict(self, x_test, numpy_output=True):

        if(type(x_test)==torch.Tensor):

            predicted = self.forward(x_test.float())

        else:

            predicted = self.forward(torch.tensor(x_test).float())


        if numpy_output:

            predicted = predicted.detach().numpy()

        return predicted

    def update_loss(self):

        self.loss = self.loss_fn(self.predict(self.X, numpy_output=False), self.y)



class DNN(nn.Module):

    def __init__(self,
                 n_dim=1,
                 dropout_prob=0.0,
                 dropout_active=False,
                 num_layers=2,
                 num_hidden=200,
                 output_size=1,
                 activation="Tanh",
                 mode="Regression"
                 ):

        super(DNN, self).__init__()

        self.n_dim          = n_dim
        self.num_layers     = num_layers
        self.num_hidden     = num_hidden
        self.mode           = mode
        self.activation     = activation
        self.device         = torch.device('cpu') # Make this an option
        self.output_size    = output_size
        self.dropout_prob   = dropout_prob
        self.dropout_active = dropout_active
        self.model          = build_architecture(self)


    def fit(self, X, y, learning_rate=1e-3, loss_type="MSE", batch_size=100, num_iter=500, verbosity=False):


        if self.n_dim!=X.shape[1]:

            self.n_dim   = X.shape[1]
            self.model   = build_architecture(self)

        self.X           = torch.tensor(X.reshape((-1, self.n_dim))).float()
        self.y           = torch.tensor(y).float()

        loss_dict        = {"MSE": torch.nn.MSELoss}

        self.loss_fn     = loss_dict[loss_type](reduction='mean')
        self.loss_trace  = []

        batch_size       = np.min((batch_size, X.shape[0]))

        optimizer        = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for _ in range(num_iter):

            batch_idx = np.random.choice(list(range(X.shape[0])), batch_size )

            y_pred    = self.model(self.X[batch_idx, :])

            self.model.zero_grad()

            optimizer.zero_grad()                   # clear gradients for this training step

            self.loss = self.loss_fn(y_pred.reshape((batch_size, 1)), self.y[batch_idx].reshape((batch_size, 1)))

            self.loss.backward(retain_graph=True)   # backpropagation, compute gradients
            optimizer.step()

            self.loss_trace.append(self.loss.detach().numpy())

            if verbosity:

                print("--- Iteration: %d \t--- Loss: %.3f" % (_, self.loss.item()))


    def predict(self, X, numpy_output=True):

        X = torch.tensor(X.reshape((-1, self.n_dim))).float()

        if numpy_output:

            prediction = self.model(X).detach().numpy()

        else:

            prediction = self.model(X)


        return prediction


    def update_loss(self):

        self.loss = self.loss_fn(self.predict(self.X, numpy_output=False), self.y)



class DNN_uncertainty_wrapper():

    def __init__(self, model, mode="exact", damp=1e-4, order=1):

        self.model            = model
        self.IF               = influence_function(model, train_index=list(range(model.X.shape[0])),
                                                   mode=mode, damp=damp, order=order)
        self.LOBO_residuals   = []

        for k in range(len(self.IF)):
            #print(k)
            perturbed_models  = perturb_model_(self.model, self.IF[k])

            ####
            #perturbed_models  = DNN(**params)
            #perturbed_models.fit(np.delete(model.X, k, axis=0).detach().numpy(),
            #                     np.delete(model.y.detach().numpy(), k, axis=0), **train_params)
            ####

            self.LOBO_residuals.append(np.abs(np.array(self.model.y[k]).reshape(-1, 1) - np.array(perturbed_models.predict(model.X[k, :])).T))

            del perturbed_models

        self.LOBO_residuals   = np.squeeze(np.array(self.LOBO_residuals))


    def predict(self, X_test, coverage=0.95):

        self.variable_preds   = []
        num_samples           = np.array(X_test).shape[0]

        for k in range(len(self.IF)):
            #print(k)
            perturbed_models  = perturb_model_(self.model, self.IF[k])

            ####
            #perturbed_models  = DNN(**params)
            #perturbed_models.fit(np.delete(model.X, k, axis=0).detach().numpy(),
            #                     np.delete(model.y.detach().numpy(), k, axis=0), **train_params)
            ####

            self.variable_preds.append(perturbed_models.predict(X_test).reshape((-1,)))

            del perturbed_models

        self.variable_preds   = np.array(self.variable_preds)

        y_upper               = np.quantile(self.variable_preds + np.repeat(self.LOBO_residuals.reshape((-1, 1)), num_samples, axis=1), 1 - (1-coverage)/2, axis=0, keepdims=False)
        y_lower               = np.quantile(self.variable_preds - np.repeat(self.LOBO_residuals.reshape((-1, 1)), num_samples, axis=1), (1-coverage)/2, axis=0, keepdims=False)

        y_pred                = self.model.predict(X_test).reshape((-1,))
        #R                     = np.repeat(self.LOBO_residuals.reshape((-1, 1)), num_samples, axis=1)
        #V                     = np.abs(np.repeat(y_pred.reshape((-1, 1)), num_samples, axis=1) - self.variable_preds)

        #CI_limit              = np.quantile(V + R, 1 - (1-coverage)/2, axis=0, keepdims=False)

        #y_upper               = y_pred + CI_limit
        #y_lower               = y_pred - CI_limit

        return y_pred, y_lower, y_upper


def Deep_ensemble(X_train, y_train, X_test, params, n_ensemble=5, train_frac=0.8):

    DEmodels = [DNN(**params) for _ in range(n_ensemble)]
    n_data   = X_train.shape[0]
    y_preds  = []

    for _ in range(n_ensemble):

        indexs   = np.random.choice(list(range(n_data)), int(np.floor(n_data * train_frac)), replace=False)

        DEmodels[_].fit(X_train[indexs, :], y_train[indexs])
        y_preds.append(DEmodels[_].predict(X_test).reshape((-1,)))

    y_pred   = np.mean(np.array(y_preds), axis=0)
    y_std    = np.std(np.array(y_preds), axis=0)

    return y_pred, y_std


class MCDP_DNN(DNN):

    def __init__(self,
                 dropout_prob=0.5,
                 dropout_active=True,
                 n_dim=1,
                 num_layers=2,
                 num_hidden=200,
                 output_size=1,
                 activation="ReLU",
                 mode="Regression"):

        super(MCDP_DNN, self).__init__()

        self.dropout_prob   = dropout_prob
        self.dropout        = nn.Dropout(p=dropout_prob)
        self.dropout_active = True


    def forward(self, X):

        _out= self.dropout(self.model(X))

        return _out


    def predict(self, X, alpha=0.1, MC_samples=100):

        z_c         = st.norm.ppf(1-alpha/2)
        X           = torch.tensor(X.reshape((-1, self.n_dim))).float()
        samples_    = [self.forward(X).detach().numpy() for u in range(MC_samples)]
        pred_sample = np.concatenate(samples_, axis=1)
        pred_mean   = np.mean(pred_sample, axis=1)
        pred_std    = z_c * np.std(pred_sample, axis=1)

        return pred_mean, pred_std


class GCN_base(torch.nn.Module):
    def __init__(self, hidden_channels, num_features):
        super(GCN_base, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_features = num_features
        #self.conv1 = GCNConv(data.num_features, hidden_channels)
        self.conv1 = GCNConv(num_features, hidden_channels)
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

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features):
        super(GCN, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_features = num_features
        self.model = GCN_base(self.hidden_channels, self.num_features)

    def fit(self, data, update_label, weight, learning_rate=1e-3, num_iter=50):
        self.data      = data
        #self.X         = torch.tensor(data.x[data.train_mask].reshape((-1, self.inputDim))).float()
        self.y         = update_label[data.train_mask]
        self.weight    = weight
        self.loss_fn   = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.decay     = 5e-4
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=self.decay)

        losses = []
        test_losses = []
        valid_losses = []
        for epoch in range(num_iter):
            self.model.zero_grad()
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)

            self.loss = self.loss_fn(out[data.train_mask], update_label[data.train_mask])
            test_loss = self.loss_fn(out[data.test_mask], update_label[data.test_mask])
            valid_loss = self.loss_fn(out[data.val_mask], update_label[data.val_mask])
            losses.append(self.loss.item())
            test_losses.append(test_loss.item())
            valid_losses.append(valid_loss.item())

            self.loss.backward(retain_graph=True)   # backpropagation, compute gradients
            optimizer.step()

            if epoch % 200 == 0:
                print(f'Epoch: {epoch:03d}, Training Loss: {self.loss:.4f}, Test Loss: {test_loss:.4f}, Valid Loss: {valid_loss:.4f}')

        plt.plot(losses,'-')
        plt.plot(test_losses,'-')
        plt.plot(valid_losses,'-')
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train','Test', 'Valid'])
        plt.title('Train vs Test vs Valid Losses with cost sensitive loss')


    def predict(self, data):

        out = self.model(data.x, data.edge_index)
        #detailed_out = out.tolist()
        #print("len of detailed_out: {}".format(len(detailed_out)))
        pred = out.argmax(dim=1)
        #print("len of pred:{}".format(len(pred)))
        #return pred
        return out


    def update_loss(self):

        self.loss = self.loss_fn(self.predict(self.data), self.y)


class GCN_uncertainty_wrapper():
    def __init__(self, model, mode="exact_gcn", damp=1e-4, order=1):
        self.model = model
        training = []
        for i, flag in enumerate(model.data.train_mask.tolist()):
            if flag:
                training.append(i)
        #print("training index:{}".format(training))

        self.IF = influence_function(model, train_index=training, mode=mode, damp=damp, order=order)

        self.LOBO_residuals = []

        for k in range(len(self.IF)):
            #print(k)
            perturbed_models = perturb_model_(self.model, self.IF[k])

            ####
            #perturbed_models  = DNN(**params)
            #perturbed_models.fit(np.delete(model.X, k, axis=0).detach().numpy(),
            #                     np.delete(model.y.detach().numpy(), k, axis=0), **train_params)
            ####
            perturbed_models_prediction = perturbed_models.predict(model.data)
            sample_k_prob = perturbed_models_prediction[k].detach().numpy()
            sample_k_pred = sample_k_prob[1]
            #self.LOBO_residuals.append(np.abs(np.array(self.model.y[k]) - np.array(perturbed_models.predict(model.X[k, :])).T))
            #print("model y at k:{}, probability of sample k:{}, and prediction of sample k:{}, resuduals:{}"
            #      .format(self.model.y[k], sample_k_prob, sample_k_pred, np.abs(np.array(self.model.y[k]) - sample_k_pred)))
            self.LOBO_residuals.append(np.abs(np.array(self.model.y[k]) - sample_k_pred))

            del perturbed_models

        self.LOBO_residuals = np.squeeze(np.array(self.LOBO_residuals))


    def predict(self, data, update_label, coverage=0.95):

        self.variable_preds_test = []
        self.variable_preds_valid = []

        test_index = []
        for i, flag in enumerate(data.test_mask.tolist()):
            if flag:
                test_index.append(i)

        val_index = []
        for i, flag in enumerate(data.val_mask.tolist()):
            if flag:
                val_index.append(i)

        num_test_samples = len(test_index)
        num_valid_samples = len(val_index)

        for k in range(len(self.IF)):
            #print(k)
            perturbed_models = perturb_model_(self.model, self.IF[k])

            ####
            #perturbed_models  = DNN(**params)
            #perturbed_models.fit(np.delete(model.X, k, axis=0).detach().numpy(),
            #                     np.delete(model.y.detach().numpy(), k, axis=0), **train_params)
            ####
            test_pred = perturbed_models.predict(data)[data.test_mask]
            test_pred_num = test_pred.detach().numpy()
            self.variable_preds_test.append(test_pred_num[:,1])

            valid_pred = perturbed_models.predict(data)[data.val_mask]
            valid_pred_num = valid_pred.detach().numpy()
            self.variable_preds_valid.append(valid_pred_num[:,1])

            del perturbed_models

        self.variable_preds_test   = np.array(self.variable_preds_test)
        self.variable_preds_valid  = np.array(self.variable_preds_valid)
        #print("variable preds:{}, and shape:{}".format(self.variable_preds, self.variable_preds.shape))
        #print("LOBO_residuals:{}, and shape:{}".format(self.LOBO_residuals, self.LOBO_residuals.shape))
        #print("coverage alpha:{}".format(coverage))

        y_upper_test             = np.quantile(self.variable_preds_test + np.repeat(self.LOBO_residuals.reshape((-1, 1)), num_test_samples, axis=1), 1 - (1-coverage)/2, axis=0, keepdims=False)
        y_lower_test             = np.quantile(self.variable_preds_test - np.repeat(self.LOBO_residuals.reshape((-1, 1)), num_test_samples, axis=1), (1-coverage)/2, axis=0, keepdims=False)
        y_prob_test              = self.model.predict(data)[data.test_mask].detach().numpy()
        y_pred_label_test        = np.argmax(y_prob_test, axis=1)
        y_label_test             = update_label[data.test_mask]

        y_upper_valid            = np.quantile(self.variable_preds_valid + np.repeat(self.LOBO_residuals.reshape((-1, 1)), num_valid_samples, axis=1), 1 - (1-coverage)/2, axis=0, keepdims=False)
        y_lower_valid            = np.quantile(self.variable_preds_valid - np.repeat(self.LOBO_residuals.reshape((-1, 1)), num_valid_samples, axis=1), (1-coverage)/2, axis=0, keepdims=False)
        y_prob_valid             = self.model.predict(data)[data.val_mask].detach().numpy()
        y_pred_label_valid       = np.argmax(y_prob_valid, axis=1)
        y_label_valid            = update_label[data.val_mask]

        return  y_label_test, y_prob_test, y_pred_label_test, y_lower_test, y_upper_test, y_label_valid, y_prob_valid, y_pred_label_valid, y_lower_valid, y_upper_valid
