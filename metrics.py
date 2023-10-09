#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

def acc_metric(pred,labels):
    bsize = pred.shape[0]
    pred_ = pred > 0.5
    acc = np.sum(pred_ == labels) 
    # import pdb;pdb.set_trace()
    acc = acc * 1.0 / bsize
    return acc

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

def confusion_matrics(labels,preds):
    # import pdb;pdb.set_trace()
    fpr, tpr, thresholds = roc_curve(labels, preds)
    precision, recall, th = precision_recall_curve(labels, preds)
    auc = roc_auc_score(labels,preds)
    # p = labels.sum()
    # n = labels.shape().sum() - p
    # import pdb;pdb.set_trace()
    try:
        return auc,precision[np.where(th>0.5)[0][0]],recall[np.where(th>0.5)[0][0]]
    except:
        # import pdb;pdb.set_trace()
        print("precision 0")
        return auc,0.1,0.1
    
    