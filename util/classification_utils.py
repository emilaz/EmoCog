#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Evals import get_f1, get_precision_recall, get_f1_from_pr
from sklearn.metrics import roc_curve, precision_recall_curve, fbeta_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np


# In[ ]:



"""""
Finds otimal true probability threshold, depending on either PR curve or ROC curve
Input: Given classifier, sklearn.cv for partitioning, (train) data, desired curve type (PR vs ROC) bool
Output: Optimal threshold for classifying
"""
def get_optimal_threshold(classifier,x,y, go_after_pr = False): #are we optimizing for f1? or tpr-fpr?
    probas_ = classifier.predict_proba(x)
    # Compute ROC curve
    #this returns different tpr/fpr for different decision thresholds
    if go_after_pr:
        pre, rec, thresholds = precision_recall_curve(y,probas_[:,1])
        f1 = get_f1_from_pr(pre,rec)
        optimal_idx = np.argmax(f1)
    else:
        fpr, tpr, thresholds = roc_curve(y, probas_[:, 1])
        optimal_idx = np.argmax(tpr - fpr)

    return thresholds[optimal_idx]



"""
Returns the median area under curve score, given a cv-partitioning and hyperparas
Input: Classifier, sklearn CV partitioning, data pair, curve type (PR vs. ROC) bool
Output: Median AUC
"""
def get_auc_score(classifier,cv,x,y, go_after_pr = False): #are we optimizing for f1? or tpr-fpr?
    aucs = []
    for train, test in cv.split(x, y):
        classifier.fit(x[train], y[train])
        probas_ = classifier.predict_proba(x[test])
        # Compute ROC curve
        #this returns different tpr/fpr for different decision thresholds
        if go_after_pr:
            aucs.append(average_precision_score(y[test],probas_[:,1]))
        else:
            aucs.append(roc_auc_score(y[test],probas_[:,1]))

    #now that we have this, what was the median AUC score?
    return np.median(aucs)

"""
Given a classifier and data x, return predictions
Input: Sklearn trained classifier, data x, given threshold
Output: Label Predictions
"""
def get_prediction(classifier,x,thresh):
    y_pred = (classifier.predict_proba(x)[:,1]>=thresh).astype(bool)
    return y_pred

"""
Fits a classifier to given train data and returns predcitions. Optionally also predicts for test data
Input: Classifier, threshold, train data, optional: test dat
"""
# def fit_predict(classifier, thresh, x,y,x_ev=None,y_ev=None):
#     classifier.fit(x,y)
#     y_pred = get_prediction(classifier,x,thresh) # predict on whole train set
#     if x_ev is None:
#         return y_pred
#     else:
#         y_pred_ev = get_prediction(classifier,x_ev,thresh) #same for ev set
#         return y_pred, y_pred_ev

"""
NOTE: What is c,g here??
Fills a pd dataframe after a) finding the optimal threshold for a train set, predicting on train and test set and then calculating scores
Input: Dataframe df, dataframe row to fill, Classifier to use, sklearn CV, train data, test data.
"""
def fit_predict_eval_fill(df,idx,classifier,cv,x,y,x_ev,y_ev):
    thresh = get_optimal_threshold(classifier, cv, x, y) # get threshold using cv
    y_pred,y_pred_ev = fit_predict(classifier, x, y, x_ev, y_ev, thresh) # using that threshold, get predictions and f1 score
    f1_tr=get_f1(y_pred,y) # calculate f1 scores for prediction on train set
    f1_ev=get_f1(y_pred_ev,y_ev)
    prec_tr,recall_tr = get_precision_recall(y_pred,y)
    prec_ev,recall_ev = get_precision_recall(y_pred_ev,y_ev)
    results_df.loc[idx] = [c,g,thresh,f1_tr,prec_tr,recall_tr,f1_ev,prec_ev,recall_ev]



"""
Return the best hyperparas from a df, given an indicator which criteria to use
Input: DF with results, criteria (normally AVGPR or something)
Output: Set of best hyperparas
"""

def get_best_hyperparas(df,col):
    pos = df[col].idxmax()
    best_row=df.loc[pos] # get the row with highest ev score
    hypers =best_row.values[:-1]
    return hypers

