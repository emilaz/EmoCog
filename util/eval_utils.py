#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import warnings


# In[17]:


"""
Function to get TP,FP,TN,FN
Input: Predictions, Ground truth
Output: True Positives, FP, TN, FN
"""
def get_pos_and_negs(pred,true):
    tp = sum(pred[pred == 1] == true[pred == 1])
    fp = sum(pred[pred == 1] != true[pred == 1])
    tn = sum(pred[pred == 0] == true[pred == 0])
    fn = sum(pred[pred == 0] != true[pred == 0])
    return tp,fp,fn,tn
    
"""
Function to get rates of TP etc.
Input: Predictions, Ground truth
Output: True Positive Rate, FPR, TNR, FNR
"""
def get_rates(pred,true):
    tp,fp,fn,tn = get_pos_and_negs(pred,true)
    tpr = tp/float(len(true[true==1]))
    fpr = fp/float(len(true[true==0]))
    tnr = tn/float(len(true[true==0]))
    fnr = fn/float(len(true[true==1]))
    return tpr,fpr,tnr,fnr

"""
Input: Predictions, Ground truth
Output: Precision, Recall
"""
def get_precision_recall(pred,true):
    tp,fp,fn,tn = get_pos_and_negs(pred,true)
    precision = tp/max(1.,float(tp+fp))
    recall = tp/max(1.,float(tp+fn))
    return precision,recall

"""
Input: Predictions, Ground truth
Outpu: F1 Score
"""
def get_f1(pred,true):
    precision,recall = get_precision_recall(pred,true)
    if precision+recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)
    sk_ver = f1_score(true,pred)
    if np.any(sk_ver != f1):
        raise ArithmeticError('Seems like my method was wrong? Check this')
    return f1

"""
Calculates F1 scores from PR arrays
Input: Precision array, Recall array
Output:  F1 array
"""


def get_f1_from_pr(precision,recall):
    ret = np.zeros(len(precision))
    numer = 2*precision*recall
    denom = precision+recall
    well_defined = denom != 0 #if precision or recall is zero, we return zero
    if np.any(well_defined == False):
        warnings.warn('Precision and Recall were zero for at least one entry.'
                      'Setting the respective F1 scores to zero.')
    ret[well_defined]= numer[well_defined]/denom[well_defined]
    #this is not clean, but servers as sanity check for above method
    f1_arr = 2*precision*recall/np.maximum(1e-6*np.ones(len(precision)),precision+recall)
    if not np.all(ret == f1_arr):
        raise ArithmeticError('Something is wrong in the calculation of the F1 score.')
    return ret

"""
Calculates Accuracy
Input: Predictions, Ground truth
Output: Accuracy
"""
def get_acc(pred,true):
    acc = sum(pred==true)/float(len(true))
    return acc

"""
Input: Predictions, Ground truth
Output: Accuracy, Precision, Recall, F1
"""
def get_all_metrics(pred,true):
    pr,re = get_precision_recall(pred,true)
    f1 = get_f1(pred,true)
    acc = get_acc(pred,true)
    return ('Accuracy',acc),('Precision',pr),('Recall',re),('F1',f1)

