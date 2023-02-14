from collections import defaultdict
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv #GATConv
import math
import matplotlib.pyplot as plt
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch_geometric.datasets import Planetoid, CitationFull, FacebookPagePage, Amazon
from torch_geometric.transforms import NormalizeFeatures

from sklearn.metrics import recall_score, accuracy_score, f1_score
from models.static import *
from utils.performance import *

from utils.preprocessing import *

import random
import sys

def ece_calibration_interval(result, prediction, label, bin_num):
    assert len(result) == len(prediction)
    prob = []
    for i, pred_label in enumerate(prediction):
        prob.append(result[i][pred_label])
    #print(len(prob))

    sample_bin = defaultdict(list)
    conf_bin = defaultdict(list)
    for i, p in enumerate(prob):
        idx = idx_bin(bin_num, p)
        conf_bin[idx].append(p)
        sample_bin[idx].append(i)

    assert len(conf_bin) == len(sample_bin)

    ece = 0
    output_confi = []
    output_acc = []
    output_majo = []
    output_mino = []
    for i in range(bin_num):
        if i not in conf_bin:
            continue
        #print(conf_bin[i])
        conf_bin[i].sort()
        min_p, max_p = range_bin(conf_bin[i])
        confi = conf(conf_bin[i])
        accu = acc(sample_bin[i], prediction, label)
        num = len(sample_bin[i])
        groud_truth_0, groud_truth_1 = count(sample_bin[i], label)
        ece += abs(confi - accu) * num
        output_confi.append(confi)
        output_acc.append(accu)
        output_majo.append(groud_truth_0)
        output_mino.append(groud_truth_1)
        #print("bin idx:{}, confidence: {:.4f}, and acc: {:.4f}, [{:.4f}, {:.4f}], [0:{}, 1:{}]".format(i, confi, accu, min_p, max_p, groud_truth_0, groud_truth_1))


    print("ECE is: {:.4f}".format(ece/len(prob)))

def ece_calibration_sample(result, prediction, label, bin_num):
    '''
    result is the detailed probability
    prediction is the predicted label for this specific sample
    label is the ground truth label
    '''
    assert len(result) == len(prediction)
    prob = []
    for i, pred_label in enumerate(prediction):
        prob.append(result[i][pred_label])
    # print(len(prob))

    sample_prob = {i:p for i, p in enumerate(prob)}
    sort_prob = {k: v for k, v in sorted(sample_prob.items(), key=lambda item: item[1])}

    sample_bin = defaultdict(list)
    conf_bin = defaultdict(list)

    avg_sample = math.floor(len(prediction)/bin_num)

    for i, k in enumerate(sort_prob):
        idx = fre_bin(avg_sample, bin_num, i)
        conf_bin[idx].append(sort_prob[k])
        sample_bin[idx].append(k)

    assert len(conf_bin) == len(sample_bin)

    ece = 0
    output_confi = []
    output_acc = []
    output_majo = []
    output_mino = []
    for i in range(bin_num):
        if i not in conf_bin:
            continue
        min_p, max_p = range_bin(conf_bin[i])
        confi = conf(conf_bin[i])
        accu = acc(sample_bin[i], prediction, label)
        num = len(sample_bin[i])
        groud_truth_0, groud_truth_1 = count(sample_bin[i], label)
        #print(sample_bin[i])
        ece += abs(confi - accu) * num
        output_confi.append(confi)
        output_acc.append(accu)
        output_majo.append(groud_truth_0)
        output_mino.append(groud_truth_1)
        # comment for less
        #print("bin idx:{}, confidence: {:.4f}, and acc: {:.4f}, [{:.4f}, {:.4f}], [0:{}, 1:{}]".format(i, confi, accu, min_p, max_p, groud_truth_0, groud_truth_1))
    print("ECE is: {:.4f}".format(ece/len(prob)))

def train_GCN(model, data, update_label, weight, optimizer, criterion, learning_rate=1e-3, num_iter=50):
    model.train()
    for epoch in range(num_iter):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        loss = criterion(out[data.train_mask], update_label[data.train_mask])
        test_loss = criterion(out[data.test_mask], update_label[data.test_mask])
        valid_loss = criterion(out[data.val_mask], update_label[data.val_mask])

        loss.backward(retain_graph=True)
        optimizer.step()

        if (epoch+1) % 500 == 0:
            print(f'Epoch: {epoch+1:03d}, Training Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Valid Loss: {valid_loss:.4f}')

def train(GJ_model, data, update_label, sample_weight, learning_rate=1e-3, num_iter=50, alpha=0.1):
    loss_fn   = torch.nn.CrossEntropyLoss(weight=sample_weight)
    decay     = 5e-4
    optimizer = torch.optim.Adam(GJ_model.model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=decay)

    original_losses = []
    uncertainty_losses = []
    integrated_losses = []
    for epoch in range(num_iter):
        optimizer.zero_grad()
        out = GJ_model.model.predict(data)

        #loss_cn = loss_fn(out[data.train_mask], update_label[data.train_mask])
        #test_loss_cn = loss_fn(out[data.test_mask], update_label[data.test_mask])
        valid_loss_cn = loss_fn(out[data.val_mask], update_label[data.val_mask])

        y_label_test, y_prob_test, y_pred_label_test, y_lower_test, y_upper_test, y_label_valid, y_prob_valid, y_pred_label_valid, y_lower_valid, y_upper_valid = GJ_model.predict(data, update_label, coverage=.90)
        y_acc_valid = (y_lower_valid + y_upper_valid)/2
        y_prob_max_valid = np.max(y_prob_valid, axis=1)
        loss_uncertainty = np.sum(abs(y_prob_max_valid - y_acc_valid))/len(y_acc_valid)

        loss = (1 - alpha) * valid_loss_cn + alpha * loss_uncertainty
        original_losses.append(valid_loss_cn.item())
        uncertainty_losses.append(loss_uncertainty.item())
        integrated_losses.append(loss.item())

        loss.backward(retain_graph=True)   # backpropagation, compute gradients
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Original loss: {valid_loss_cn:.4f}, Uncertainty Loss: {loss_uncertainty:.4f}, integrated Loss: {loss:.4f}')

            if epoch >= 500:
                train_acc, test_acc, val_acc, train_rec, test_rec, val_rec, f1_macro, f1_micro, f1, out, pred, update_label = test(GJ_model_16.model,data)
                print()
                print(f'Train Accuracy: {train_acc:.4f} \t Test Accuracy: {test_acc:.4f} \t Valid Accuracy: {val_acc:.4f}')
                print(f'Train Recall: {train_rec:.4f} \t Test Recall: {test_rec:.4f} \t Valid Recall: {val_rec:.4f}')
                print(f'Test Macro F1: {f1_macro:.4f} \t Test Micro F1: {f1_micro:.4f} ')
                print("F1:{}".format(f1))

                model_result = out[data.test_mask].tolist()
                model_argmax = pred[data.test_mask].tolist()
                label = update_label[data.test_mask].tolist()
                majo_prob, majo_result, majo_label, mino_prob, mino_result, mino_label = splitCate(model_result, model_argmax, label)

                ece_calibration_interval(model_result, model_argmax, label, 20)
                ece_calibration_sample(model_result, model_argmax, label, 20)
                ece_calibration_sample(majo_prob, majo_result, majo_label, 20)
                ece_calibration_sample(mino_prob, mino_result, mino_label, 20)

def test_gcn(model, data):
    out = model(data.x, data.edge_index)
    #detailed_out = out.tolist()
    #print("len of detailed_out: {}".format(len(detailed_out)))
    pred = out.argmax(dim=1)
    #print("len of pred:{}".format(len(pred)))
    train_acc = accuracy_score(update_label[data.train_mask].tolist(), pred[data.train_mask].tolist())
    test_acc = accuracy_score(update_label[data.test_mask].tolist(), pred[data.test_mask].tolist())
    val_acc = accuracy_score(update_label[data.val_mask].tolist(), pred[data.val_mask].tolist())

    train_recall = recall_score(update_label[data.train_mask].tolist(), pred[data.train_mask].tolist())
    test_recall = recall_score(update_label[data.test_mask].tolist(), pred[data.test_mask].tolist())
    val_recall = recall_score(update_label[data.val_mask].tolist(), pred[data.val_mask].tolist())

    f1_macro = f1_score(update_label[data.test_mask].tolist(), pred[data.test_mask].tolist(), average='macro')
    f1_micro = f1_score(update_label[data.test_mask].tolist(), pred[data.test_mask].tolist(), average='micro')
    f1 = f1_score(update_label[data.test_mask].tolist(), pred[data.test_mask].tolist(), average=None)

    return train_acc, test_acc, val_acc, train_recall, test_recall, val_recall, f1_macro, f1_micro, f1, out, pred, update_label

def test(model, data):
    out = model.predict(data)
    #detailed_out = out.tolist()
    #print("len of detailed_out: {}".format(len(detailed_out)))
    pred = out.argmax(dim=1)
    #print("len of pred:{}".format(len(pred)))
    train_acc = accuracy_score(update_label[data.train_mask].tolist(), pred[data.train_mask].tolist())
    test_acc = accuracy_score(update_label[data.test_mask].tolist(), pred[data.test_mask].tolist())
    val_acc = accuracy_score(update_label[data.val_mask].tolist(), pred[data.val_mask].tolist())

    train_recall = recall_score(update_label[data.train_mask].tolist(), pred[data.train_mask].tolist())
    test_recall = recall_score(update_label[data.test_mask].tolist(), pred[data.test_mask].tolist())
    val_recall = recall_score(update_label[data.val_mask].tolist(), pred[data.val_mask].tolist())

    f1_macro = f1_score(update_label[data.test_mask].tolist(), pred[data.test_mask].tolist(), average='macro')
    f1_micro = f1_score(update_label[data.test_mask].tolist(), pred[data.test_mask].tolist(), average='micro')
    f1 = f1_score(update_label[data.test_mask].tolist(), pred[data.test_mask].tolist(), average=None)

    return train_acc, test_acc, val_acc, train_recall, test_recall, val_recall, f1_macro, f1_micro, f1, out, pred, update_label