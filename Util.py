
# coding: utf-8

# In[5]:


import numpy as np
from sklearn.decomposition import PCA
from Evals import get_f1, get_precision_recall
from sklearn.metrics import roc_curve


# In[3]:


class ClassificationUtils:
    def get_optimal_threshold(classifier,cv,x,y):
        optimal_threshs = []
        for train, test in cv.split(x, y):
            classifier.fit(x[train], y[train])
            probas_ = classifier.predict_proba(x[test])
            # Compute ROC curve
            #this returns different tpr/fpr for different decision thresholds
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            optimal_threshs.append(optimal_threshold)

        #now that we have this, what was the median best threshold?
        return np.median(optimal_threshs)

    def get_prediction(classifier,x,thresh):
        y_pred = (classifier.predict_proba(x)[:,1]>thresh).astype(bool)
        return y_pred
    
    def fit_predict(classifier,x,y,x_ev,y_ev,thresh):
        classifier.fit(x, y) # fit the classifier
        y_pred = ClassificationUtils.get_prediction(classifier,x,thresh) # predict on whole train set
        y_pred_ev = ClassificationUtils.get_prediction(classifier,x_ev,thresh) #same for ev set
        return y_pred, y_pred_ev
    
    def get_best_hyperparas_results(df):
        pos = df['F1 Score Ev'].idxmax()
        best_row=df.loc[pos] # get the row with highest ev score
        return best_row       
    
    def fit_predict_eval_fill(df,idx,classifier,cv,x,y,x_ev,y_ev):
        thresh = ClassificationUtils.get_optimal_threshold(classifier, cv, x, y) # get threshold using cv
        y_pred,y_pred_ev = ClassificationUtils.fit_predict(classifier, x, y, x_ev, y_ev, thresh) # using that threshold, get predictions and f1 score
        f1_tr=get_f1(y_pred,y) # calculate f1 scores for prediction on train set
        f1_ev=get_f1(y_pred_ev,y_ev)
        prec_tr,recall_tr = get_precision_recall(y_pred,y)
        prec_ev,recall_ev = get_precision_recall(y_pred_ev,y_ev)
        results_df.loc[idx] = [c,g,thresh,f1_tr,prec_tr,recall_tr,f1_ev,prec_ev,recall_ev]


# In[4]:


class FeatureUtils:
    def standardize(data,ax=0):
        data_mean = np.mean(data,axis=ax)
        data_dem = data-data_mean
        std = np.std(data,axis=ax)
        data_stand = data_dem/std
        return data_stand

    #this function caps at 150Hz, then bins the data in a logarithmic fashion to account for smaller psd values in higher freqs
    def bin_psd(fr,psd):
        fr_trun=fr[fr<=150]
        fr_total=len(fr_trun)
        fr_bins=np.arange(int(np.log2(max(fr_trun))+1))
        #truncate everythin above 150Hz
        psd=psd[:,fr<=150]
        psd_bins=np.zeros((psd.shape[0],len(fr_bins)))
        prev=0
        #these are the general upper limits. they don't give info where in fr/psd these frequencies acutally are!
        max_psd_per_bin=np.exp2(fr_bins).astype('int')
        #hence we need this method:
        prev=0
        limits=np.zeros((max_psd_per_bin.shape[0],2),dtype='int')
        for en,b in enumerate(max_psd_per_bin):
            if en==0:
                arr=np.where((fr_trun >=prev)&(fr_trun<=b))[0]
            else:
                arr=np.where((fr_trun >prev)&(fr_trun<=b))[0]
            check=np.array([min(arr),max(arr)])
            limits[np.log2(b).astype('int')]=check
            prev=b
        prev=0
        for b in fr_bins:
            if (b==fr_bins[-1] or limits[b][1]>=fr_total):
                psd_bins[:,b]+=np.sum(psd[:,limits[b,0]:],axis=1)
            else:
                psd_bins[:,b]=np.sum(psd[:,limits[b,0]:limits[b,1]+1],axis=1)
        return fr_bins, psd_bins


    def get_no_comps(data,expl_var_lim):
        comps=min(100,min(data.shape))
        pca=PCA(n_components=comps)
        pca.fit(data)
        tot=0
        for idx,c in enumerate(pca.explained_variance_ratio_):
            tot+=c
            if tot*100>expl_var_lim:
                return idx+1
        return pca.n_components_

