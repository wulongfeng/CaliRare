from collections import defaultdict
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv #GATConv
import math

from sklearn.metrics import recall_score, accuracy_score, f1_score

import random
import sys

def mask2set(mask_list):
    id_set = set()
    for i, v in enumerate(mask_list):
        if v == True:
            id_set.add(i)
    return id_set


def dataset_split(seed, data, num_sample_train, num_test, num_val, num_sample):
    random.seed(seed)
    train = set()
    test = set()
    val = set()
    label = data.y.tolist()
    label_idx = defaultdict(list)

    for i, l in enumerate(label):
        label_idx[l].append(i)

    for label in label_idx:
        corpus = label_idx[label]
        cache = set()
        while len(cache) < num_sample_train:
            r = random.randint(0, len(corpus) -1)
            cache.add(corpus[r])
        train.update(cache)
    #print("adding {} nodes for training:{}".format(len(train), train))

    while len(test) < num_test:
        r = random.randint(0, len(data.y) -1)
        if r not in train:
            test.add(r)
    #print("adding {} nodes for testing:{}".format(len(test), test))

    while len(val) < num_val:
        r = random.randint(0, len(data.y) -1)
        if r not in train and r not in test:
            val.add(r)
    #print("adding {} nodes for validation:{}".format(len(val), val))

    train_mask, test_mask, val_mask =  [False] * num_sample, [False] * num_sample, [False] * num_sample
    for i in train:
        train_mask[i] = True
    for i in test:
        test_mask[i] = True
    for i in val:
        val_mask[i] = True

    data.train_mask = torch.BoolTensor(train_mask)
    data.test_mask = torch.BoolTensor(test_mask)
    data.val_mask = torch.BoolTensor(val_mask)
    return train, test, val

def more_train_withSplit(data, seed, new_len, train, test, val):
    random.seed(seed)
    data_label = data.y.tolist()
    label_statis = defaultdict(list)
    for i, label in enumerate(data_label):
        label_statis[label].append(i)
    #print(label_statis)

    add_idx = []


    for label in label_statis:
        corpus = label_statis[label]
        cache = set()
        while len(cache) < new_len:
            r = random.randint(0, len(corpus) -1)
            if corpus[r] not in train and corpus[r] not in test and corpus[r] not in val:
                cache.add(corpus[r])
        #print(cache)
        add_idx.extend(cache)
    #print("adding {} nodes:{}".format(len(add_idx), add_idx))
    for i in add_idx:
        data.train_mask[i] = True

    print("the length for training set is: {}".format(data.train_mask.sum()))


def more_train(data, random_seed, new_len, idx_start, idx_end):
    random.seed(random_seed)
    data_label = data.y.tolist()
    label_statis = defaultdict(list)
    for i, label in enumerate(data_label):
        label_statis[label].append(i)
    #print(label_statis)
    add_idx = []

    for label in label_statis:
        corpus = label_statis[label]
        cache = set()
        while len(cache) < new_len:
            r = random.randint(0, len(corpus) - 1)
            if corpus[r]> idx_start and corpus[r]< idx_end:
                cache.add(corpus[r])
        #print(cache)
        add_idx.extend(cache)
    #print("adding {} nodes:{}".format(len(add_idx), add_idx))
    for i in add_idx:
        data.train_mask[i] = True

    print("the length for training set is: {}".format(data.train_mask.sum()))

def more_train_v2(data, random_seed, new_len, train_set, test_set, val_set):
    random.seed(random_seed)
    data_label = data.y.tolist()
    label_statis = defaultdict(list)
    for i, label in enumerate(data_label):
        label_statis[label].append(i)
    #print(label_statis)
    add_idx = []

    for label in label_statis:
        corpus = label_statis[label]
        cache = set()
        while len(cache) < new_len:
            r = random.randint(0, len(corpus) - 1)
            if corpus[r] not in train_set and corpus[r] not in test_set and corpus[r] not in val_set:
                cache.add(corpus[r])
        #print(cache)
        add_idx.extend(cache)
    #print("adding {} nodes:{}".format(len(add_idx), add_idx))
    for i in add_idx:
        data.train_mask[i] = True

    print("the length for training set is: {}".format(data.train_mask.sum()))


def acc(idx, prediction, label):
    if len(idx) == 0:
        return 0

    correct = 0
    for i in idx:
        if prediction[i] == label[i]:
            correct += 1
    return correct/len(idx)

def count(idx, label):
    update_y_count = defaultdict(int)
    for i in idx:
        update_y_count[label[i]] +=1
    #print(update_y_count)
    return update_y_count[0], update_y_count[1]

def conf(conf_list):
    if len(conf_list) == 0:
        return 0
    else:
        return sum(conf_list)/len(conf_list)

def range_bin(conf_list):
    if len(conf_list) == 0:
        print("empty list")
    return conf_list[0], conf_list[-1]

def idx_bin(bin_num, p):
    if p == 1:
        print("the last interval")
        return bin_num - 1
    interval = 1/bin_num
    return math.floor(p/interval)

def fre_bin(avg_sample, bin_num, order):
    if order >= bin_num * avg_sample:
        return bin_num - 1
    return math.floor(order/avg_sample)

def showFloatList(list_num):
    new_list = []
    for num in list_num:
        new_list.append("{:.4f}".format(num))
    print("\t".join(new_list))

def showIntList(list_num):
    #print("\t".join(str(list_num)))
    print(*list_num, sep='\t')

def splitCate(prob, model_result, label):
    mino_prob = []
    mino_result = []
    mino_label = []

    majo_prob = []
    majo_result = []
    majo_label = []

    for i in range(len(label)):
        if label[i] == 0:
            majo_prob.append(prob[i])
            majo_result.append(model_result[i])
            majo_label.append(label[i])
        else:
            mino_prob.append(prob[i])
            mino_result.append(model_result[i])
            mino_label.append(label[i])
    assert len(majo_prob) == len(majo_result)
    assert len(majo_prob) == len(majo_label)
    assert len(mino_prob) == len(mino_result)
    assert len(mino_prob) == len(mino_label)

    return majo_prob, majo_result, majo_label, mino_prob, mino_result, mino_label

def prob_mino(prob, label):
    prob = prob.tolist()

    mino_prob = []
    for i, i_label in enumerate(label):
        if i_label == 1:
            mino_prob.append(prob[i])
    return mino_prob

def prob_majo(prob, label):
    prob = prob.tolist()

    majo_prob = []
    for i, i_label in enumerate(label):
        if i_label == 0:
            majo_prob.append(prob[i])
    return majo_prob

