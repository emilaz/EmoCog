
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import supereeg as se
import numpy as np
import pandas as pd
from nilearn import plotting as ni_plt
import pdb
from Evals import *

from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


# In[4]:


class ClassificationVis:
    
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
        tp,fp,tn,fn = get_pos_and_negs(pred,true)
        rates = np.array([tp,fp,fn,tn]).reshape((2,2))
        df_cm = pd.DataFrame(rates, index = ['Pred Happy','Pred Not Happy'],columns = ['True Happy','True Not Happy'])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True,fmt='g')
        plt.show()
        

    def plot_roc(x,y,classifier,  title):
        cv = StratifiedKFold(n_splits=5,random_state=0)
        tprs = []
        aucs = []
        fpr_interval = np.linspace(0, 1, 100)
        i = 0
        for train, test in cv.split(x, y):
            probas_ = classifier.fit(x[train], y[train]).predict_proba(x[test])
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
        plt.title(title)# w/ RBF kernel, d=%2d' %gamma)
        plt.legend(loc="lower right")
        plt.show()



# In[ ]:


# class LabelVis:
#     # #plot the nan ratio
#     def nan_ratio():
        
# br=np.unique(np.array(test.pred_bin, dtype='float'), return_counts=True)
# sum_nans=np.sum(br[1][2:])
# #print(sum_nans)
# vals=([str(br[0][0]),str(br[0][1]),str(br[0][2])],[br[1][0],br[1][1],sum_nans])
# print(vals[0],vals[1])
# plt.bar(vals[0],vals[1])
# plt.title("Occurences of 'Happy'/'Not Happy'/'N/A' predictions in %ds of data" % (stop-start))
# plt.xlabel('Prediction')
# plt.ylabel('Occurences')


# In[2]:


class BrainVis:

    def plot_brain(chan_labels,num_grid_chans=64,colors=list()):
        mni_coords_fullfile='/data2/users/stepeter/mni_coords/cb46fd46/cb46fd46_MNI_atlasRegions.xlsx'
        'Plots ECoG electrodes from MNI coordinate file'
        #Example code to run it: 
             #import sys
             #sys.path.append('/home/stepeter/AJILE/stepeter_sandbox/ECoG_Preprocessing')
             #from plot_ecog_electrodes_mni import *

             #mni_coords_fullfile='/data2/users/stepeter/mni_coords/a0f66459/a0f66459_MNI_atlasRegions.xlsx'
             #plot_ecog_electrodes_mni_from_file_and_labels(mni_coords_fullfile,chan_num_min=-1,chan_num_max=-1,num_grid_chans=64)

        #NOTE: A warning may pop up the first time running it, leading to no output. Rerun this function, and the plots should appear.

        #Load in MNI file
        mni_file = pd.read_excel(mni_coords_fullfile, delimiter=",")


        #Create dataframe for electrode locations
        locs=mni_file.loc[mni_file['Electrode'].isin(chan_labels)][['X coor', 'Y coor', 'Z coor']]
        print(locs.shape)

        #Label strips/depths differently for easier visualization (or use defined color list)
        if len(colors)==0:
            for s in range(locs.shape[0]):
                if s>=num_grid_chans:
                    colors.append('r')
                else:
                    colors.append('b')

        #Plot the result
        ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                               node_kwargs={'alpha': 0.5, 'edgecolors': None},
                               node_size=10, node_color=colors)

