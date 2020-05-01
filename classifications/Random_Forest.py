import sys
sys.path.append('..')


import vis.classification_vis as cvis
from data_creation.data_provider import *
import util.classification_utils as util
import util.data_utils as dutil


import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import os

from itertools import product
from multiprocessing import Pool


def train_and_save_best_classifier(results,x,y,configs):
    best_n, best_d, best_f = util.get_best_hyperparas(results,'AVG PR')
    classifier = RandomForestClassifier(n_estimators=int(best_n), max_depth=int(best_d), max_features=int(best_f),random_state=0) #init classifier
    classifier.fit(x,y)
    best_thr = util.get_optimal_threshold(classifier, x, y, go_after_pr=True) # get threshold using cv (on whole dataset)
    dutil.save_classifier(classifier, best_thr, configs,'RF')
    
    

# Run classifier with cross-validation
def calc_results_and_save(x, y, configs):
    cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)
    #dataframe for saving results
    results = pd.DataFrame(columns=('Number Estimators','Max Depth','Max Features','AVG PR'))#,'AUC ROC'))
    #do random search on parameters
    est = np.random.choice(np.arange(130)[5:],25)
    max_d = np.random.choice(np.arange(60)[1:],25)
    max_f = np.random.choice(np.arange(min(x.shape))[1:],25)

    #Search for best hyperpara combo with cross validation
    for idx,(c,d,f) in enumerate(zip(est,max_d,max_f)):
        classifier = RandomForestClassifier(n_estimators=c, max_depth=d, max_features=f,random_state=0) #init classifier
        auc_pr = util.get_auc_score(classifier, cv, x, y, go_after_pr=True)
        #auc_roc =util.get_auc_score(classifier, cv, x, y, go_after_pr=False)
        results.loc[idx] = [c,d,f,auc_pr]
        print('Number Estimators= %d, Max Depth = %d, Max Feat = %d, AUC PR %.3f' % (c,d,f,auc_pr))
    # if shuff:
    #     dutil.save_results(results,configs,'RF_Shuffle') # save results for later use
    else:
        dutil.save_results(results,configs,'RF') # save results for later use      
    return results


def vis_results(x,y, x_ev, y_ev, configs):
    ###using hyperpara found, evaluate and get pretty plots
    #get f1 scores on whole training set
    classifier, best_thr = dutil.load_classifier(configs,'RF')
    y_pred = util.get_prediction(classifier,x,best_thr)
    y_pred_ev = util.get_prediction(classifier,x_ev,best_thr)

    #draw pretty plots
    title = dutil.generate_filename(configs)
    cvis.score_heatmap(y_pred, y, 'Metrics Train Set '+title)
    cvis.score_heatmap(y_pred_ev, y_ev, 'Metrics Test Set '+title)
    cvis.conf_mat(y_pred,y, 'Confusion Matrix Train Set '+title)
    cvis.conf_mat(y_pred_ev,y_ev,'Confusion Matrix Test Set '+title)

    cvis.plot_roc(x,y,classifier, 'Random Forest ROC Train Set '+title)
    cvis.plot_roc(x_ev,y_ev,classifier,  'Random Forest ROC Test Set '+title)
    cvis.plot_pr_curve(x,y,classifier, 'Random Forest PR curve Train Set '+title)
    cvis.plot_pr_curve(x_ev,y_ev,classifier, 'Random Forest PR curve Test Set '+title)

    
def do_all(file, cut, reload= False):
    provider = DataProvider()
    configs = dutil.generate_configs_from_file(file, cut)
    print(configs)
    x,y,x_ev,y_ev = provider.get_data(configs, reload=True)
    print(x.shape,y.shape,x_ev.shape,y_ev.shape)
    if reload:
        try:
            res = dutil.load_results(configs,'RF')
        # this assumes we just want to visualize. If you want to actually calulcate here, do differently
        except FileNotFoundError as e:
            print(configs)
            print(e)
            return
    else:
        res = calc_results_and_save(x,y,configs)
        # best classifier is the one with best AUC (PR or ROC)
        train_and_save_best_classifier(res,x,y,configs)

    vis_results(x,y, x_ev, y_ev, configs)
    
def randomize_labels(y):
    ones = int(np.sum(y))
    #where?
    fill_ones = np.random.choice(len(y),ones, replace=False)
    ret = np.empty(len(y))
    ret[:]=0
    ret[fill_ones]=1
    return ret
    

if __name__ == '__main__':
    files = [f for f in os.listdir('/home/emil/EmoCog/data/new_labels/train_test_datasets') if 'af859cc5' in f and '[3, 4, 5]' in f]
    cuts = [.3,.5,.7]
    reload = [False]
    all_elements = [files,cuts, reload]

    file_cut_combos = []
    for allel in product(*all_elements):
        file_cut_combos += [allel]

    pool = Pool(8)
    yass = pool.starmap(do_all,file_cut_combos)

    del(pool)
    del(yass)

