
# coding: utf-8

# In[6]:


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

def get_acc(pred,true):
    acc = sum(pred==true)/float(len(true))
    return acc

def get_all_metrics(pred,true):
    pr,re = get_precision_recall(pred,true)
    f1 = get_f1(pred,true)
    acc = get_acc(pred,true)
    return ('Accuracy',acc),('Precision',pr),('Recall',re),('F1',f1)

def get_conf_mat(pred,true):
    tp,fp,tn,fn = get_pos_and_negs(pred,true)
    rates = np.array([tp,fp,fn,tn]).reshape((2,2))
    df_cm = pd.DataFrame(rates, index = ['Pred Happy','Pred Not Happy'],columns = ['True Happy','True Not Happy'])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True,fmt='g')

