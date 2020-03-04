import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from matplotlib import colors
from Evals import *
from util.classification_utils import get_optimal_threshold, get_prediction
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy import interp

import pandas as pd
import numpy as np

def bar_plot_svc(scores_tr,scores_ev,gammas,thresholds = None):
    bar_width = 0.35
    plt.xticks(np.arange(0,20)+bar_width/2,range(0,20))
    bar1 = plt.bar(gammas,scores_tr,bar_width,label='Train',alpha=.5,color='b')
    bar2 = plt.bar(gammas+bar_width,scores_ev,bar_width,label='Ev',alpha=.5,color='r')
    plt.hlines(.5,0,max(gammas)+1,linestyle='dashed',alpha=.2)
    #plt.plot(np.arange(0,15),[.5]*15,'--',color='k')
    plt.xlabel('Dim')
    plt.ylabel('F1 Score')
    plt.xlim(0,max(gammas)+1)
    plt.ylim(0,1)
    plt.legend()

    # Add thresholds above the two bar graphs
    if thresholds:
        for idx,rects in enumerate(zip(bar1,bar2)):
            higher=np.argmax([rects[0].get_height(),rects[1].get_height()])
            rect=rects[higher]
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()-(higher/2.), height, '%.2f' % thresholds[idx], ha='center', va='bottom')      
    plt.show()

def plot_svc(scores_tr,scores_ev,hyper,label='Hyperpara'):
    plt.scatter(hyper,scores_tr,label='Train')
    plt.scatter(hyper,scores_ev, label='Ev')
    plt.legend()
    plt.title('F1 Score')
    plt.xlabel(label)
    plt.ylabel('F1 - Score')
    plt.ylim(0,1)
    plt.show()

def conf_mat(pred,true):
    tp,fp,fn,tn = get_pos_and_negs(pred,true)
    rates = np.array([tp,fp,fn,tn]).reshape((2,2))
    df_cm = pd.DataFrame(rates, index = ['Pred Happy','Pred Not Happy'],columns = ['True Happy','True Not Happy'])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True,fmt='g',annot_kws={"size": 26})
    plt.show()


def plot_roc(x,y,classifier,  title): #the classifier has to be pretrained here!!
    cv = StratifiedKFold(n_splits=10, shuffle=False)
    tprs = []
    aucs = []
    fpr_interval = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(x, y):
        probas_ = classifier.predict_proba(x[test])
        # Compute ROC curve and area the curve
        #this returns different tpr/fpr for different decision thresholds
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        curr_interp = interp(fpr_interval, fpr, tpr)
        tprs.append(curr_interp)
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        i += 1


    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
              label='Chance', alpha=.8)


    mean_interpol_tpr = np.mean(tprs, axis=0)
    mean_interpol_auc = auc(fpr_interval, mean_interpol_tpr)
    std_auc = np.std(aucs)
    plt.plot(fpr_interval, mean_interpol_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_interpol_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_interpol_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_interpol_tpr - std_tpr, 0)
    plt.fill_between(fpr_interval, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_pr_curve(x, y, classifier, title):
    y_probs = classifier.predict_proba(x)
    avg_p  = average_precision_score(y,y_probs[:,1]) #get the average precision score
    precision, recall, _ = precision_recall_curve(y, y_probs[:,1])
    #step_kwargs = ({'step': 'post'}
    #       if 'step' in signature(plt.fill_between).parameters
    #       else {})
    step_kwargs = ({'step': 'post'})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title + ', AP={0:0.2f}'.format(
              avg_p))
    plt.show()
    
    
def _background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]
    
    
    
def print_results(df_res):
    pretty = df_res.style.apply(_background_gradient,
               cmap='PuBu',
               m=df_res.min().min(),
               M=df_res.max().max(),
               low=0,
               high=1)
    display(pretty)
    
    
    
"""
Function to plot an average ROC curve, given many random trials of shuffled labels.
Currentlz onlz works with Random Forest.
Input: pandas df, where each row contains the best hyperparas, the PR result and the shuffled y, y_ev. x, x_ev
"""


def plot_roc_random(df, title, train = True): 
    tprs = []
    aucs = []
    fpr_interval = np.linspace(0, 1, 100)
    i = 0
    for idx, row in df.iterrows():
        classifier = RandomForestClassifier(n_estimators=int(row['Number Estimators']), max_depth=int(row['Max Depth']), max_features=int(row['Max Features']), random_state=0)
        y = row['y']
        y_ev = row['y_ev']
        x = row['x']
        x_ev = row['x_ev']
        classifier.fit(x,y)
        if train:
            probas_ = classifier.predict_proba(x)
            # Compute ROC curve and area the curve
            #this returns different tpr/fpr for different decision thresholds
            fpr, tpr, thresholds = roc_curve(y, probas_[:, 1])
        else:
            probas_ = classifier.predict_proba(x_ev)
            # Compute ROC curve and area the curve
            #this returns different tpr/fpr for different decision thresholds
            fpr, tpr, thresholds = roc_curve(y_ev, probas_[:, 1])
        curr_interp = interp(fpr_interval, fpr, tpr)
        tprs.append(curr_interp)
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        i += 1


    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
              label='Chance', alpha=.8)


    mean_interpol_tpr = np.mean(tprs, axis=0)
    mean_interpol_auc = auc(fpr_interval, mean_interpol_tpr)
    std_auc = np.std(aucs)
    plt.plot(fpr_interval, mean_interpol_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_interpol_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_interpol_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_interpol_tpr - std_tpr, 0)
    plt.fill_between(fpr_interval, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def conf_mat_random(df, train):
    count = 0
    res = np.zeros(4)
    pr = 0
    rec = 0
    f1 = 0
    for idx, row in df.iterrows():
        classifier = RandomForestClassifier(n_estimators=int(row['Number Estimators']), max_depth=int(row['Max Depth']), max_features=int(row['Max Features']), random_state=0)
        y = row['y']
        y_ev = row['y_ev']  
        x = row['x']
        x_ev = row['x_ev']
        classifier.fit(x,y)
        best_thr = get_optimal_threshold(classifier, x, y, go_after_pr=True) # get threshold using cv (on whole dataset)
        y_pred = get_prediction(classifier, x, best_thr) # using that threshold, get predictions and f1 score
        y_pred_ev = get_prediction(classifier, x_ev, best_thr) # using that threshold, get predictions and f1 score
        
        if train:
            res += np.array(get_pos_and_negs(y_pred,y))
            p,r = get_precision_recall(y_pred,y)
            f = get_f1(y_pred,y)
            f1 += f
            pr += p
            rec += r
        else:
            res += np.array(get_pos_and_negs(y_pred_ev,y_ev))
            p,r = get_precision_recall(y_pred_ev,y_ev)
            f = get_f1(y_pred_ev,y_ev)
            f1 += f
            pr += p
            rec += r
        count += 1
    if train:
        print('Train')
    else:
        print('Eval')
    print('Avg. Precision {0:.3f}, Avg. Recall {1:.3f}, Avg. F1 {2:.3f}'.format(pr/count,rec/count,f1/count))
    df_cm = pd.DataFrame(res.reshape(2,2)/count, index = ['Pred Happy','Pred Not Happy'],columns = ['True Happy','True Not Happy'])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True,fmt='g',annot_kws={"size": 26})
    plt.show()
    return pr/count,rec/count,f1/count

 
    
def happy_ratio_random(df):
    new_df = pd.DataFrame(columns = ['Happy', 'Train', 'Test'])

    for idx, row in df.iterrows():
        new_df.loc[2*idx] = [True,np.sum(row['y']),np.sum(row['y_ev'])]
        new_df.loc[2*idx+1] = [False,len(row['y'])-np.sum(row['y']),len(row['y_ev'])-np.sum(row['y_ev'])]

    llel = new_df.melt(id_vars = 'Happy',value_vars = ['Train','Test'])

    llel['value'] = llel['value'].astype('int')
    llel['Happy'] = llel['Happy'].astype('bool')

    sns.set()

    sns.violinplot(x='variable',data=llel,y='value',hue='Happy',split=True, hue_order =[True,False], palette = {True:'blue', False:'red'} )
    plt.title('Violin Plot of Happy/Not Happy distribution')
    plt.ylabel('Counts')
    plt.xlabel('Datset')
    plt.show()
