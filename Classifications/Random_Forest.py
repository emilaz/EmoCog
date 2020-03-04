#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import vis.classification_vis as cvis
from Evals import *
from Data_Provider import *
import util.classification_utils as util
import util.data_utils as dutil
import util.label_utils as lutil


import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import os

from itertools import product
from multiprocessing import Pool


# In[3]:


def train_and_save_best_classifier(results,x,y,configs):
    best_n, best_d, best_f = util.get_best_hyperparas(results,'AVG PR')
    classifier = RandomForestClassifier(n_estimators=int(best_n), max_depth=int(best_d), max_features=int(best_f),random_state=0) #init classifier
    classifier.fit(x,y)
    best_thr = util.get_optimal_threshold(classifier, x, y, go_after_pr=True) # get threshold using cv (on whole dataset)
    dutil.save_classifier(classifier, best_thr, configs,'RF')
    
    

# Run classifier with cross-validation
def calc_results_and_save(x, y, configs, shuff):
    cv = StratifiedKFold(n_splits=10, shuffle=shuff, random_state=1)
    #dataframe for saving results
    results = pd.DataFrame(columns=('Number Estimators','Max Depth','Max Features','AVG PR'))#,'AUC ROC'))
    #do random search on parameters
    est = np.random.choice(np.arange(130)[5:],50)
    max_d = np.random.choice(np.arange(60)[1:],50)
    max_f = np.random.choice(np.arange(min(x.shape))[1:],50)

    #Search for best hyperpara combo with cross validation
    for idx,(c,d,f) in enumerate(zip(est,max_d,max_f)):
        classifier = RandomForestClassifier(n_estimators=c, max_depth=d, max_features=f,random_state=0) #init classifier
        auc_pr = util.get_auc_score(classifier, cv, x, y, go_after_pr=True)
        #auc_roc =util.get_auc_score(classifier, cv, x, y, go_after_pr=False)
        results.loc[idx] = [c,d,f,auc_pr]
        print('Number Estimators= %d, Max Depth = %d, Max Feat = %d, AUC PR %.3f' % (c,d,f,auc_pr))
    if shuff:
        dutil.save_results(results,configs,'RF_Shuffle') # save results for later use
    else:
        dutil.save_results(results,configs,'RF') # save results for later use      
    return results


def vis_results(x,y, x_ev, y_ev, configs):
    ###using hyperpara found, evaluate and get pretty plots
    #get f1 scores on whole training set
    classifier, best_thr = dutil.load_classifier(configs,'RF')
    y_pred = util.get_prediction(classifier,x,best_thr)
    y_pred_ev = util.get_prediction(classifier,x_ev,best_thr)
    f1_tr=get_f1(y_pred,y) # calculate f1 scores for prediction on train set
    f1_ev=get_f1(y_pred_ev,y_ev)
    prec_tr,recall_tr = get_precision_recall(y_pred,y)
    prec_ev,recall_ev = get_precision_recall(y_pred_ev,y_ev)

    df_res = pd.DataFrame(index =['Train','Eval'],columns = ['Precision','Recall','F1']).astype('float')
    df_res.loc['Train'] = [prec_tr, recall_tr,f1_tr]
    df_res.loc['Eval'] = [prec_ev, recall_ev,f1_ev]
    cvis.print_results(df_res)
    #draw pretty plots
    cvis.conf_mat(y_pred,y)
    cvis.conf_mat(y_pred_ev,y_ev)

    cvis.plot_roc(x,y,classifier, 'Random Forest ROC on Train Set')
    cvis.plot_roc(x_ev,y_ev,classifier,  'Random Forest ROC on Eval Set')
    cvis.plot_pr_curve(x,y,classifier, 'Random Forest Pr-Re curve on Train Set')
    cvis.plot_pr_curve(x_ev,y_ev,classifier, 'Random Forest ROC on Eval Set')

    
def do_all(file, cut, shuffled=False, random = False, reload= False):
# def do_all(data, cut, shuffled=False, random = True):
    provider = DataProvider()
    configs = dutil.generate_configs_from_file(file, cut)
    print(configs)
    x,y,x_ev,y_ev = provider.get_data(configs)
    print(x.shape,y.shape,x_ev.shape,y_ev.shape)
    #if random:
    if False:
        print('yes, random')
        np.random.seed()
        y = randomize_labels(y)
        y_ev = randomize_labels(y_ev)
    if reload:
        res = dutil.load_results(configs,'RF')
    else:
        res = calc_results_and_save(x,y,configs,shuffled)
    
    train_and_save_best_classifier(res,x,y,configs)

    res = calc_results_and_save(x,y,shuffled)
#     #### get best resul on train set
#     pos = res['AVG PR'].idxmax()
#     best_row=list(res.loc[pos].values) # get the row with highest ev score
#     best_row.append(y)
#     best_row.append(y_ev)
#     return best_row
    ####
    vis_results(x,y, x_ev, y_ev, configs)
    
def randomize_labels(y):
    ones = int(np.sum(y))
    #where?
    fill_ones = np.random.choice(len(y),ones, replace=False)
    ret = np.empty(len(y))
    ret[:]=0
    ret[fill_ones]=1
    return ret
    
    


# In[4]:


files = [f for f in os.listdir('/home/emil/OpenMindv2/data/postprocessing') if 'shuffle_True' in f and '3' in f]
cuts = [.05,.1,.2,.3]
# shuffled =[False]
all_elements = [files,cuts]
# #all_elements = [[bla],cuts]

file_cut_combos = []
for allel in product(*all_elements):
    file_cut_combos+=[allel]


# In[5]:


files


# In[ ]:


# do_all(files[2],.1)

pool = mp.Pool(7)
yass = pool.starmap(do_all,file_cut_combos)

del(pool)
del(yass)
