from collections import defaultdict
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid, CitationFull, FacebookPagePage, Amazon
from torch_geometric.transforms import NormalizeFeatures

from sklearn.metrics import recall_score, accuracy_score, f1_score
from models.static import *
from utils.preprocessing import *
from utils.evaluate import *

import random
import sys
import argparse
import os
import logging

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Calibration for Rare categories',
        usage='test.py [<args>] [-h | --help]'
    )
    parser.add_argument('--dataset', type=str, default='Cora', help="name of the dataset")
    parser.add_argument('-weight', type=float, default=1.0, help="cost sensitive")
    parser.add_argument('-minority_label', type=int, default=0,
                        help="the index of class chosen to be the minority class")
    parser.add_argument('-seed', default=0, type=int, help="random seed")
    parser.add_argument('-added_label_rate', default=0, type=int, help="samples added to the training set per class")
    parser.add_argument('-alpha', type=float, default=0.9, help="the coverage parameter")
    parser.add_argument('-lam', type=float, default=0.1,
                        help="the tradeoff parameter between calibration and classification")
    return parser.parse_args(args)

def main(args):
    print("dataset:{}, type:{}".format(args.dataset, type(args.dataset)))
    print("weight:{}, type:{}".format(args.weight, type(args.weight)))
    print("minority_label:{}, type:{}".format(args.minority_label, type(args.minority_label)))
    print("seed:{}, type:{}".format(args.seed, type(args.seed)))
    print("added_label_rate:{}, type:{}".format(args.added_label_rate, type(args.added_label_rate)))
    print("alpha:{}, type:{}".format(args.alpha, type(args.alpha)))
    print("lam:{}, type:{}".format(args.lam, type(args.lam)))

    if args.dataset in ['Cora', 'PubMed', 'CiteSeer']:
        dataset = Planetoid(root='data/Planetoid', name=args.dataset, transform=NormalizeFeatures())
    elif args.dataset == 'DBLP':
        dataset = CitationFull(root='data/DBLP', name='DBLP', transform=NormalizeFeatures())
    elif args.dataset == 'FaceBook':
        dataset = FacebookPagePage(root='data/FacebookPagePage', transform=NormalizeFeatures())
    elif args.dataset == 'Amazon_Photo':
        dataset = Amazon(root='data_torch1.4/Amazon_photo', name='Photo', transform=NormalizeFeatures())

    print(50*'=')
    data = dataset[0]
    print(data)
    if args.dataset in ['DBLP', 'FaceBook', 'Amazon_Photo']:
        samples = data.num_nodes
        train_set, test_set, val_set = dataset_split(args.seed, data, 20, 1000, 500, samples)
    elif args.dataset in ['Cora', 'PubMed', 'CiteSeer']:
        train_set = mask2set(data.train_mask)
        test_set = mask2set(data.test_mask)
        val_set = mask2set(data.val_mask)
    print(f'Number of training nodes:{data.train_mask.sum()}\t'
          f'Number of testing nodes:{data.test_mask.sum()}\t'
          f'Number of validation nodes:{data.val_mask.sum()}')

    # pre-processing for label information
    label = data.y.tolist()
    statis = defaultdict(int)
    for l in label:
        statis[l] += 1
    print("detailed info about category:{}".format(statis))
    label_rate = args.added_label_rate
    if label_rate != 0:
        print("Adding more samples.....")
        #random_seed = int(param[5])
        #add_train = int(param[6])
        ##idx_start, idx_end
        if args.dataset in ['Cora', 'PubMed', 'CiteSeer']:
            more_train_v2(data, args.seed, args.added_label_rate, train_set, test_set, val_set)
        else:
            more_train_withSplit(data, args.seed, args.added_label_rate, train_set, test_set, val_set)

    print(f'Number of training nodes:{data.train_mask.sum()}\t'
          f'Number of testing nodes:{data.test_mask.sum()}\t'
          f'Number of validation nodes:{data.val_mask.sum()}')
    # updating label (y) information
    update_y = []
    for l in label:
        if l == args.minority_label:
            update_y.append(1)
        else:
            update_y.append(0)

    update_label = torch.LongTensor(update_y)
    update_y_count = defaultdict(int)
    for l in update_y:
        update_y_count[l] += 1
    print("detailed info about category after updating:{}".format(update_y_count))

    # training model
    #weight = float(param[2])
    weights = torch.tensor([args.weight, 1], dtype=torch.float32)
    weights = 1.0 / weights
    final_weights = weights / weights.sum()
    weights_16_1 = torch.FloatTensor(final_weights)
    print("weight:{}".format(weights_16_1))

    num_fea = data.num_features
    print(num_fea)
    params        = dict({"hidden_channels": 16, "num_features":num_fea})
    GCN_model_16 = GCN(**params)

    print("training starts..")
    GCN_model_16.fit(data, update_label, weights_16_1, num_iter=2000)

    train_acc, test_acc, val_acc, train_rec, test_rec, val_rec, f1_macro, out, pred, update_label = test_gcn(GCN_model_16.model, data, update_label)
    print()
    print(f'Train Accuracy: {train_acc:.4f} \t Test Accuracy: {test_acc:.4f} \t Valid Accuracy: {val_acc:.4f} \n'
          f'Train Recall: {train_rec:.4f} \t Test Recall: {test_rec:.4f} \t Valid Recall: {val_rec:.4f} \n'
          f'Test Macro F1: {f1_macro:.4f}')

    model_result = out[data.test_mask].tolist()
    model_argmax = pred[data.test_mask].tolist()
    label = update_label[data.test_mask].tolist()
    majo_prob, majo_result, majo_label, mino_prob, mino_result, mino_label = splitCate(model_result, model_argmax, label)

    ece_calibration_interval(model_result, model_argmax, label, 20)
    ece_calibration_sample(model_result, model_argmax, label, 20)
    ece_calibration_sample(majo_prob, majo_result, majo_label, 20)
    ece_calibration_sample(mino_prob, mino_result, mino_label, 20)


    GJ_model_16 = GCN_uncertainty_wrapper(GCN_model_16, order=1, damp=1e-2)

    y_label_test, y_prob_test, y_pred_label_test, y_lower_test, y_upper_test, y_label_valid, y_prob_valid, y_pred_label_valid, y_lower_valid, y_upper_valid = GJ_model_16.predict(data, update_label, coverage=args.alpha)
    print()
    print("training with individual calibration..")
    train(GJ_model_16, data, update_label, weights_16_1, learning_rate=1e-3, num_iter=200, alpha=args.lam)

    train_acc, test_acc, val_acc, train_rec, test_rec, val_rec, f1_macro, out, pred, update_label = test(GJ_model_16.model, data, update_label)
    print()
    print(f'Train Accuracy: {train_acc:.4f} \t Test Accuracy: {test_acc:.4f} \t Valid Accuracy: {val_acc:.4f} \n'
          f'Train Recall: {train_rec:.4f} \t Test Recall: {test_rec:.4f} \t Valid Recall: {val_rec:.4f} \n'
          f'Test Macro F1: {f1_macro:.4f}')

    model_result = out[data.test_mask].tolist()
    model_argmax = pred[data.test_mask].tolist()
    label = update_label[data.test_mask].tolist()
    majo_prob, majo_result, majo_label, mino_prob, mino_result, mino_label = splitCate(model_result, model_argmax, label)

    ece_calibration_interval(model_result, model_argmax, label, 20)
    ece_calibration_sample(model_result, model_argmax, label, 20)
    ece_calibration_sample(majo_prob, majo_result, majo_label, 20)
    ece_calibration_sample(mino_prob, mino_result, mino_label, 20)


if __name__ == "__main__":
    main(parse_args())

