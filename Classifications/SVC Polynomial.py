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


# Run classifier with cross-validation
def calc_results_and_save(x, y, configs,shuff):
    cv = StratifiedKFold(n_splits=10,shuffle=shuff,random_state=1)
    results = pd.DataFrame(columns=('C','Gamma','Degree','AVG PR'))#,'AUC ROC'))
    degree=np.random.choice(np.arange(10)[1:],50)
    cs = np.random.uniform(0,4,50)
    gammas = 2**(np.random.choice(18,50)-15.)

    for idx,(deg,c,g) in enumerate(zip(degree,cs,gammas)):
        classifier = svm.SVC(cache_size=16000, C=c,kernel='poly', probability = True,gamma =g,degree=deg,random_state=5)
        auc_pr = util.get_auc_score(classifier, cv, x, y, go_after_pr=True)
        #auc_roc =util.get_auc_score(classifier, cv, x, y, go_after_pr=False)
        results.loc[idx] = [c,g,deg, auc_pr]
        print('Gamma = %.6f, Dim = %d, C= %.2f, Avg PR %.3f' %(g,deg,c,auc_pr))
    if shuff:
        dutil.save_results(results,configs,'Poly') # save results for later use
    else:
        dutil.save_results(results,configs,'Poly_NoShuffle') # save results for later use 
    return results


def vis_results(x,y, x_ev, y_ev, hypers):
    best_c, best_g, best_d = hypers
    ###using hyperpara found, evaluate and get pretty plots
    cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
    classifier = svm.SVC(C=best_c,kernel='poly', probability = True,gamma =best_g,degree=best_d,random_state=5)
    best_thr = util.get_optimal_threshold(classifier, cv, x, y, go_after_pr=True) # get threshold using cv
    y_pred,y_pred_ev = util.fit_predict(classifier, best_thr, x, y, x_ev, y_ev) # using that threshold, get predictions and f1 score
    f1_tr=get_f1(y_pred,y) # calculate f1 scores for prediction on train set
    f1_ev=get_f1(y_pred_ev,y_ev)
    prec_tr,recall_tr = get_precision_recall(y_pred,y)
    prec_ev,recall_ev = get_precision_recall(y_pred_ev,y_ev)

    print('Training, Precision: {0:.3f}, Recall: {1:.3f}, F1: {2:.3f}'.format(prec_tr, recall_tr,f1_tr))
    print('Eval, Precision: {0:.3f}, Recall: {1:.3f}, F1: {2:.3f}'.format(prec_ev, recall_ev,f1_ev))
    cvis.conf_mat(y_pred,y)
    cvis.conf_mat(y_pred_ev,y_ev)
    plt.show()

    cvis.plot_roc(x,y,classifier, 'SVC w/ Poly Kernel, degree= {:.2f}, C= {:.2f} Gamma = {:.2f}'.format(best_d,best_c, best_g))
    cvis.plot_roc(x_ev,y_ev,classifier, 'SVC w/ Poly Kernel, degree= {:.2f}, C= {:.2f} Gamma = {:.2f}'.format(best_d,best_c, best_g))
    cvis.plot_pr_curve(x,y,classifier, 'SVC w/ Poly Kernel, degree= {:.2f}, C= {:.2f} Gamma = {:.2f}'.format(best_d,best_c, best_g))
    cvis.plot_pr_curve(x_ev,y_ev,classifier,'SVC w/ Poly Kernel, degree= {:.2f}, C= {:.2f} Gamma = {:.2f}'.format(best_d,best_c, best_g))


def do_all(file, cut, shuffled=False, random = False):
    provider = DataProvider()
    configs = dutil.generate_configs_from_file(file, cut)
    print(configs)
    x,y,x_ev,y_ev, loadings = provider.get_data(configs)
#def do_all(x,y,x_ev,y_ev, shuffled=True):
    if random:
        print('bla')
        np.random.seed()
        y = randomize_labels(y)
        y_ev = randomize_labels(y_ev)
    res = calc_results_and_save(x,y,configs,shuffled)
    #hypers = util.get_best_hyperparas(res,'AVG PR')
    #vis_results(x,y, x_ev, y_ev,hypers)
    
def randomize_labels(y):
    ones = int(np.sum(y))
    #where?
    fill_ones = np.random.choice(len(y),ones, replace=False)
    ret = np.empty(len(y))
    ret[:]=0
    ret[fill_ones]=1
    return ret
    


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


files = [f for f in os.listdir('/home/emil/EmoCog/data/new_labels/train_test_datasets')]
cuts = [.1,.2,.3]
shuffle =[False] 
all_elements = [files,cuts,shuffle]
file_cut_combos = []
for allel in product(*all_elements):
    file_cut_combos+=[allel]
    
pool = mp.Pool(6)
yass = pool.starmap(do_all,file_cut_combos)


# In[ ]:


del(pool)
del(yass)

