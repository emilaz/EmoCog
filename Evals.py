
# coding: utf-8

# In[1]:


import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[17]:


def get_pos_and_negs(pred,true):
    tp = sum(pred[pred == 1] == true[pred == 1])
    fp = sum(pred[pred == 1] != true[pred == 1])
    tn = sum(pred[pred == 0] == true[pred == 0])
    fn = sum(pred[pred == 0] != true[pred == 0])
    return tp,fp,tn,fn
    
def get_rates(pred,true):
    tp,fp,tn,fn = get_pos_and_negs(pred,true)
    tpr = tp/float(len(true[true==1]))
    fpr = fp/float(len(true[true==0]))
    tnr = tn/float(len(true[true==0]))
    fnr = fn/float(len(true[true==1]))
    return tpr,fpr,tnr,fnr

def get_precision_recall(pred,true):
    tp,fp,tn,fn = get_pos_and_negs(pred,true)
    precision = tp/max(1.,float(tp+fp))
    recall = tp/max(1.,float(tp+fn))
    return precision,recall

def get_f1(pred,true):
    precision,recall = get_precision_recall(pred,true)
    f1 = 2*precision*recall/max(1.,(precision+recall))
    return f1

def get_f1_from_pr(precision,recall):
    f1_arr = 2*precision*recall/np.maximum(np.ones(len(precision)),precision+recall)
    return f1_arr

def get_acc(pred,true):
    acc = sum(pred==true)/float(len(true))
    return acc

def get_all_metrics(pred,true):
    pr,re = get_precision_recall(pred,true)
    f1 = get_f1(pred,true)
    acc = get_acc(pred,true)
    return ('Accuracy',acc),('Precision',pr),('Recall',re),('F1',f1)

