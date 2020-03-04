import sys
sys.path.append('..')
import os

from Evals import *
from Data_Provider import *
import util.classification_utils as util
import util.data_utils as dutil

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import StratifiedKFold


from itertools import product




#Run classifier with cross-validation and plot ROC curves
def calc_results_and_save(x,y,configs, shuff):
    cv = StratifiedKFold(n_splits=10,shuffle=shuff,random_state = 1)
    results = pd.DataFrame(columns=('C','Gamma','AVG PR'))#,'AUC ROC'))
    cs = np.random.uniform(0,5,50)
    gammas = 2**(np.random.choice(18,50)-15.)

    for idx,(g,c) in enumerate(zip(gammas,cs)):
        classifier = svm.SVC(cache_size=16000,C=c,kernel='rbf', probability = True,gamma = g, random_state=5)
        auc_pr = util.get_auc_score(classifier, cv, x, y, go_after_pr=True)
        #auc_roc =util.get_auc_score(classifier, cv, x, y, go_after_pr=False)
        results.loc[idx] = [c,g,auc_pr]
        print('Gamma = %.6f, C= %2.2f, Avg PR: %.4f' % (g,c,auc_pr))
    if shuff:
        dutil.save_results(results,configs,'RBF') # save results for later use
    else:
        dutil.save_results(results,configs,'RBF_NoShuffle') # save results for later use
    return results


def vis_results(x,y, x_ev, y_ev, hypers,shuff):
    best_c, best_g = hypers
    ###using hyperpara found, evaluate and get pretty plots
    classifier = svm.SVC(C=best_c, kernel='rbf', probability=True,gamma = best_g,random_state=5)
    cv = StratifiedKFold(n_splits=10, shuffle=shuff, random_state=1)
    best_thr = util.get_optimal_threshold(classifier, cv, x, y, go_after_pr=True) # get threshold using cv
    #for thr in np.linspace(0,1,10):
    print(best_thr)
    y_pred,y_pred_ev = util.fit_predict(classifier, best_thr, x, y, x_ev, y_ev) # using that threshold, get predictions and f1 score
    f1_tr=get_f1(y_pred,y) # calculate f1 scores for prediction on train set
    f1_ev=get_f1(y_pred_ev,y_ev)
    prec_tr,recall_tr = get_precision_recall(y_pred,y)
    prec_ev,recall_ev = get_precision_recall(y_pred_ev,y_ev)

    print(prec_tr, recall_tr,f1_tr)
    print(prec_ev, recall_ev,f1_ev)

    
    ClassificationVis.conf_mat(y_pred,y)
    ClassificationVis.conf_mat(y_pred_ev,y_ev)
    plt.show()
    classifier.fit(x,y)
    ClassificationVis.plot_roc(x,y,classifier, 'SVC, RBF Kernel, C ={:.2f}, Gamma = {:.5f}'.format(best_c,best_g))
    ClassificationVis.plot_roc(x_ev,y_ev,classifier, 'SVC, RBF Kernel, C ={:.2f}, Gamma = {:.5f}'.format(best_c,best_g))
    ClassificationVis.plot_pr_curve(x,y,classifier,'SVC, RBF Kernel, C ={:.2f}, Gamma = {:.5f}'.format(best_c,best_g))
    ClassificationVis.plot_pr_curve(x_ev,y_ev,classifier,'SVC, RBF Kernel, C ={:.2f}, Gamma = {:.5f}'.format(best_c,best_g))


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
