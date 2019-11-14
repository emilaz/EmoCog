#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from Evals import get_f1, get_precision_recall, get_f1_from_pr
from sklearn.metrics import roc_curve, precision_recall_curve, fbeta_score, roc_auc_score, average_precision_score
import os
import datetime
from functools import reduce


# In[2]:


class ClassificationUtils:
    
#     def get_optimal_threshold(classifier,cv,x,y, go_after_pr = False): #are we optimizing for f1? or tpr-fpr?
#         optimal_threshs = []
#         for train, test in cv.split(x, y):
#             classifier.fit(x[train], y[train])
#             probas_ = classifier.predict_proba(x[test])
#             # Compute ROC curve
#             #this returns different tpr/fpr for different decision thresholds
#             if go_after_pr:
#                 pre, rec, thresholds = precision_recall_curve(y[test],probas_[:,1])
#                 f1 = get_f1_from_pr(pre,rec)
#                 optimal_idx = np.argmax(f1)
#             else:
#                 fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#                 optimal_idx = np.argmax(tpr - fpr)
#             optimal_threshold = thresholds[optimal_idx]
#             optimal_threshs.append(optimal_threshold)

#         #now that we have this, what was the median best threshold?
#         return np.median(optimal_threshs)

    """""
    Finds otimal true probability threshold, depending on either PR curve or ROC curve
    Input: Given classifier, sklearn.cv for partitioning, (train) data, desired curve type (PR vs ROC) bool
    Output: Optimal threshold for classifying
    """
    def get_optimal_threshold(classifier,cv,x,y, go_after_pr = False): #are we optimizing for f1? or tpr-fpr?
        classifier.fit(x,y)
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
                
        #now that we have this, what was the median best threshold?
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
    def fit_predict(classifier, thresh, x,y,x_ev=None,y_ev=None):
        classifier.fit(x,y)
        y_pred = ClassificationUtils.get_prediction(classifier,x,thresh) # predict on whole train set
        if x_ev is None:
            return y_pred
        else:
            y_pred_ev = ClassificationUtils.get_prediction(classifier,x_ev,thresh) #same for ev set
            return y_pred, y_pred_ev
    
    """
    NOTE: What is c,g here??
    Fills a pd dataframe after a) finding the optimal threshold for a train set, predicting on train and test set and then calculating scores
    Input: Dataframe df, dataframe row to fill, Classifier to use, sklearn CV, train data, test data.
    """
    def fit_predict_eval_fill(df,idx,classifier,cv,x,y,x_ev,y_ev):
        thresh = ClassificationUtils.get_optimal_threshold(classifier, cv, x, y) # get threshold using cv
        y_pred,y_pred_ev = ClassificationUtils.fit_predict(classifier, x, y, x_ev, y_ev, thresh) # using that threshold, get predictions and f1 score
        f1_tr=get_f1(y_pred,y) # calculate f1 scores for prediction on train set
        f1_ev=get_f1(y_pred_ev,y_ev)
        prec_tr,recall_tr = get_precision_recall(y_pred,y)
        prec_ev,recall_ev = get_precision_recall(y_pred_ev,y_ev)
        results_df.loc[idx] = [c,g,thresh,f1_tr,prec_tr,recall_tr,f1_ev,prec_ev,recall_ev]
        
    """
    Gives you the row with the best results, given a column specifying which metric to judge by (F1 or other)
    Input: Datafrane, Column
    Output: Row with best results
    """
    def get_best_hyperparas_results(df, col):
        pos = df[col].idxmax()
        best_row=df.loc[pos] # get the row with highest ev score
        return best_row       


# In[3]:


"""
This class is mainly concerned with saving and loading data. 
It also includes a method to set the usual hyperparameters for loading data faster (no need to calculate them again)
"""
class DataUtils:

    def load_configs():
        configs = dict()
        configs['sliding'] = False
        configs['wsize'] = 5
        configs['s_sample']= 0
        configs['e_sample']= 30000
        configs['s_sample_ev'] = 30000
        configs['e_sample_ev'] = 35000
        configs['cutoff'] = .2
        return configs
    """
    Generates a filename out of the configs
    Input: Configs
    Output: Filename
    """
    def generate_filename(configs):
        fname = 'ws_'+str(configs['wsize'])+'_str_'+str(configs['sliding'])+'_tr'+'_s_'+str(configs['s_sample'])+'_e_'+str(configs['e_sample'])+'_ev_'+'s_'+str(configs['s_sample_ev'])+'_e_'+str(configs['e_sample_ev'])+'_expvar_'+str(configs['expvar'])
        return fname
    
    """
    Generates configs from a filename
    Input: Filename
    Output: Configs
    """
    def generate_configs_from_file(file,cutoff=None):
        configs = dict()
        underscores = [pos for pos, char in enumerate(file) if char == '_']
        configs['wsize'] = int(file[underscores[0]+1:underscores[1]])
        try:
            configs['sliding'] = int(file[underscores[2]+1:underscores[3]])
        except ValueError: #not sure if 0 isn't leading to a sliding window with stride 0, hence this safety measure
            configs['sliding'] = False
        configs['s_sample'] = int(file[underscores[5]+1:underscores[6]])
        configs['e_sample'] = int(file[underscores[7]+1:underscores[8]])
        configs['s_sample_ev'] = int(file[underscores[10]+1:underscores[11]])
        configs['e_sample_ev'] = int(file[underscores[12]+1:underscores[13]])
        configs['expvar'] = int(file[underscores[14]+1:underscores[14]+3])
        if cutoff:
            configs['cutoff'] = float(cutoff)
        return configs
           
    """
    This method loads the data (train and test) into memory, given the configs. Obviously only works when data for these configs has been generated and saved some previous time
    Input: Configs
    Output: Train and test data according to these configs
    """
    def load_data_from_file(configs):
        fname = DataUtils.generate_filename(configs)
        link = '/home/emil/OpenMindv2/data/several_days/'+fname+'.hdf'
        df = pd.read_hdf(link)
        x = df['x'][0]
        y = df['y'][0]
        x_ev = df['x_ev'][0]
        y_ev = df['y_ev'][0]
        return x,y,x_ev,y_ev
    
    """
    This method saves the generated train and test data to a file, given the corresponding configs. Automatically generates an appropriate filename.
    Input: Train and test data, configs
    Output: None
    """
    def save_data_to_file(x,y,x_ev,y_ev,configs):
        fname = DataUtils.generate_filename(configs)
        link = '/home/emil/OpenMindv2/data/several_days/'+fname+'.hdf'
        #save stuff to file:
        df = pd.DataFrame(data=[[x,y,x_ev,y_ev]],columns=['x','y','x_ev','y_ev'])
        df.to_hdf(link,key='df')
        
        
    
    """
    This methods loads data given a file path
    Input: File path
    Output: x,y,x_ev,y_ev
    """
    def load_data_from_path(path):
        df = pd.read_hdf(path)
        x = df['x'][0]
        y = df['y'][0]
        x_ev = df['x_ev'][0]
        y_ev = df['y_ev'][0]
        return x,y,x_ev,y_ev
    
        
    """
    This method saves the obtained results to a file for later inspection.
    Input: Datafrme with results, config file, ML approach used
    Output: None
    """
    def save_results(df, configs,methodtype):
        fname = DataUtils.generate_filename(configs)
        fname = fname +'_cut_'+str(configs['cutoff']) #cutoff info is important for results
        link = '/home/emil/OpenMindv2/data/results/several_days/'+fname+methodtype
        df.to_hdf(link,key='df')
    
    """
    Returns a dataframe with results, given the configs and the ML-method type in question
    Input: Configs, Method
    Output: Dataframe with results
    """
    def load_results(configs,methodtype):
        fname = DataUtils.generate_filename(configs)
        fname = fname +'_cut_'+str(configs['cutoff'])
        link = '/home/emil/OpenMindv2/data/results/several_days/'+fname+methodtype
        df = pd.read_hdf(link)
        return df
        


# In[231]:


class FeatureUtils:
    
    """
    Standardizes data (demean& unit variance)
    Input: Data, standart deviation of data, mean of data
    """
    def standardize(data,std,data_mean):
        data_dem = data-data_mean[:,None]
        data_stand = data_dem/(std[:,None])
        return data_stand
    
    """
    Return indices of bad time points, based on outlier detection
    Input: Data matrix
    Output: Bad indices matrix, same length as data matrix
    """
    
    def detect_artifacts(data_matrix, std_lim=None, med_lim = None):
        #first, we only want to look at the high-freq bin:
        high_freqs = data_matrix[7::8,:]
        #next, calculate median std across each channel(should be 1 I think)
        if std_lim is None:
            std_lim = np.std(high_freqs,axis=1)
            med_lim = np.median(high_freqs,axis=1)
        #next, for each row, get all indices that are too far away from the median in one direction
        too_high = high_freqs>(med_lim+3*std_lim)[:,None] #should yield a 2D matrix
        #for each column, check if there is an entry that's too high
        too_high_idx = np.any(too_high,axis=0)
        #same for other direction
        too_low = high_freqs<(med_lim-3*std_lim)[:,None] #should yield a 2D matrix
        too_low_idx = np.any(too_low,axis=0)
        bad_idx = too_low_idx | too_high_idx #any index either too high or too low? that is a bad index
        return bad_idx, std_lim, med_lim
    
    
    def remove_artifacts(data, bad_indices):
        good_data = data[:,~bad_indices] #keep only the good indices
        return good_data
        

    #this function caps at 150Hz, then bins the data in a logarithmic fashion to account for smaller psd values in higher freqs
    """
    Key function for feature processing. Truncates frequencies above 150Hz, bins the frequencies logarithmically.
    Throws the PSD into these bins by summing all PSD that fall into a certain bin.
    Input: Frequency array, PSD array
    Output: Binned frequencies, binned PSD
    """
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
        #hence we need this method. It gives us a 2D array with the number of rows = number of bins,
        #where each row contains 2 values, upper and lower limit of that bin
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
        #now, we will the psd_bins by adding up all psd in the range defined before. The last bin gets all remaining frequencies
        #meaning, if the last bin ranges from 64-128, it'll also include everything until 150.
        for b in fr_bins:
            if (b==fr_bins[-1] or limits[b][1]>=fr_total):
                psd_bins[:,b]+=np.sum(psd[:,limits[b,0]:],axis=1)
            else:
                psd_bins[:,b]=np.sum(psd[:,limits[b,0]:limits[b,1]+1],axis=1)
        return fr_bins, psd_bins

    
    """
    Function for PCA.
    Given some minimum of explained variance of the data, return the number of components needed (at most 100).
    Input: Data, desired variance explained
    Output: Number of Components needed.
    """
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
    
    
    
    """
    Function to find the indices of channels that were good across days.
    Input: PD Series with good channel info per day
    Output: Good Indices?
    """
    def find_common_channels(channels_per_day):
        common = reduce(np.intersect1d, channels_per_day['GoodChans']) #which channels are good on all days?
        #get the indices of these common channels, for each day
        good_df = pd.DataFrame(columns = ['Day','CommonChans'])
        for idx, row in channels_per_day.iterrows():
            indices=[np.min(np.nonzero(row['GoodChans']==ch)[0]) for ch in common] #these are the indices to keep - not necessarily in correct order!
            good = np.zeros(len(row['GoodChans']),dtype='bool') #therefore, bool array for order
            good[indices] = True
            good_df.loc[idx] = [row['Day'], good]
        return good_df
    
    
    def filter_common_channels(common_df):
        good_idx = FeatureUtils.find_common_channels(common_df[['Day','GoodChans']])
        ret_df = common_df.copy()
        for idx,day in enumerate(good_idx['Day']):
            good = good_idx.loc[good_idx['Day']==day,'CommonChans'][idx]
            new_data = common_df[common_df['Day']==day]['BinnedData'][idx][good,:,:]
            ret_df.loc[common_df['Day']==day,'BinnedData'] = [new_data]
        return ret_df
        


# In[5]:


class LabelUtils:
    
    
    
    """
    Converts strings into int/np.nan. Necessary for if looking at annotations in dataframe
    Input: Data
    Output: Mal gukcen
    """
    def convert_labels_readable(annot):
        nan_indices= annot=='N/A'
        annot[annot!='Happy']=0
        annot[annot=='Happy']=1
        annot[nan_indices]=np.nan
        return annot


    
    """
    Converts labels in range [0,1] into binary Happy/Not Happy, given cutoff
    Input: Labels, cutoff
    Output: Binary Labels
    """
    
    def do_cutoff(y_in,cutoff):
        y = y_in.copy()
        y[y>cutoff] = 1
        y[y<1] = 0
        return y
    

    """
    For a given video, finds the number of frames the video is supposed to have.
    Input: The hdf dataframe from Gautham's network, the given video
    Output: Number of frames the video should have
    """
    
    def find_number_frames(df, vid, last_vid, fill_vid_after):
        next_vid = str(int(vid)+1) #which is the next number?
        len_vid = len(next_vid)
        next_vid = '0'*(4-len_vid)+next_vid
        time_vid = df[df['vid']==vid]['datetime'].iloc[0]
        try:
            time_next_vid = df[df['vid']==next_vid]['datetime'].iloc[0]
            time = datetime.datetime.strptime(time_vid, '%Y-%m-%d %H:%M:%S.%f')
            new_time = datetime.datetime.strptime(time_next_vid, '%Y-%m-%d %H:%M:%S.%f')
            elapsed = (new_time-time).total_seconds()
        except: #in case of last video
            if last_vid != vid:
                fill_vid_after[0] = True
            elapsed = 120
        elapsed_frames = int(elapsed *30)
        return elapsed_frames
    
    """
    This function distributes the given m frames along an array of size n (>m), the rest are nans
    Input: The given labels for some video, the number of frames(and therefore labels) the video should have
    Output: Array of length n with frames distributed along array (in ordered manner!)"""

    def fill_frames(actual_labels,supposed_no_labels):
        no_labels = len(actual_labels) #how many labels do we have?
        places = np.empty(supposed_no_labels) #create array of length we actually want
        places[:]=np.nan
        positions = sorted(np.random.choice(supposed_no_labels,no_labels,replace=False)) #which positions do we want to fill?
        places[positions]=actual_labels
        return places        


# In[259]:


class SyncUtils:
    
    
    """
    Given a day, finds the corresponding video file I guess
    """
    #TODO: Check if session is longer than a day
    def find_paths(patient_name,day_no):
        #create filename:
        filename = patient_name+'_fullday_'+str(day_no)+'.h5'
        path = os.path.join('/nas/ecog_project/derived/processed_ecog',patient_name,'full_day_ecog')
        ecog_path = os.path.join(path, filename)
        #check if file exists
        if not os.path.exists(ecog_path):
            print('ECoG File does not exist. Check if day or patient name is wrong.')
            return
        else: #ok nice, now get video paths
            #read in the csv file. also includes the corresponding ecog file
            vid_info = os.path.join(path, 'vid_start_end_merge.csv')
            vid_csv =pd.read_csv(vid_info)
            vid_relevant = vid_csv[vid_csv['merge_day']==int(day_no)]
            #this is a bit hacky. To get the video session that fits into the given day
            try:
                sess_start = vid_relevant[vid_relevant['filename'].str.contains('0000')].iloc[0]
            except:
                FileNotFoundError('No session start for this day. Something might be wrong in SyncUtils.')
            #what is the session name?
            sess = sess_start['filename'].split('_')[1]
            #we want something akin to this format: cb46fd46_5_imp_columns
            vid_filename = '_'.join([patient_name,sess,'imp','columns.hdf'])
            vid_path = os.path.join('/home/emil/data/hdf_data',vid_filename)
            #check if gautham file exists
            if not os.path.exists(vid_path):
                print('ECoG File does not exist. Check if day or patient name is wrong.')
                return
            return ecog_path, vid_path
        
    """
    This function finds out how many seconds into the day the given session start
    Input: Path to video session hdf file
    Output: Seconds passed since midnight until video session began"""
    def find_start_and_end_time(vid_path):
        #first_frame = pd.read_hdf(vid_path, stop = 1) #read only first line of hdf file
        store = pd.HDFStore(vid_path)
        #first, where is the start of the video, in secs?
        start_time = store.select('df',stop =1)['realtime'].iloc[0]
        start_in_secs = (start_time - start_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        #now for the end of the video. First, check if still same day
        nrows = store.get_storer('df').shape[0]    
        end_time = store.select('df', start = nrows -1, stop = nrows)['realtime'].iloc[0]
        if end_time.date() == start_time.date():
            end_in_secs = (end_time - end_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        else: #if the file goes well into the next day, then we'll just everything until midnight
            print('Current session goes into the next day, so edge case. Check if things went correctly.')
            end_in_secs = 24*3600
        store.close()
        return int(start_in_secs), int(end_in_secs)
    
    
    """
    Function to filter bad/faulty data points. The order is important, since (esp. bad feature points) indices are relative to previous filterings
    Input: x,y, indices
    Output: x,y filtered
    """
    
    def filter_data(x,y,feat_ind):
        #first, filter the NaNs on feature side out on label side as well
        bad_feat_nan = np.array(feat_ind['NaNs'])
        print(bad_feat_nan, 'indices to kill')
        bad_feat_nan_inbounds = bad_feat_nan[bad_feat_nan<len(y)] #this shouldn't do anything I think
        if(len(bad_feat_nan)>len(bad_feat_nan_inbounds)):
            print('jo diese laengen solltten gleich sein.')
        y_no_feat_nans = np.delete(y,bad_feat_nan)
        #now the artifacts
        bad_feat_artifacts = feat_ind['Artifacts'] 
        bad_feat_artifacts_inbounds = bad_feat_artifacts[bad_feat_artifacts<len(y_no_feat_nans)]
        if(len(bad_feat_artifacts)<len(bad_feat_artifacts_inbounds)):
            print('jo diese laengen sollten auch gleich sein.')
        y_no_arts = y_no_feat_nans[~bad_feat_artifacts_inbounds]
        #finally, filter out the nans on label side:
        x_ret = x[:,~np.isnan(y_no_arts)]
        y_ret = y_no_arts[~np.isnan(y_no_arts)]
        return x_ret, y_ret
        
    

