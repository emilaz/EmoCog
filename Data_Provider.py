#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


from FeatureRelated.feature_generator import Feature_generator
from LabelRelated.label_generator import Label_generator 
from FeatureRelated.feature_data_holder import FeatDataHolder
from LabelRelated.label_data_holder import LabelDataHolder
import numpy as np
import pandas as pd
from Vis import LabelVis, ClassificationVis
import util.data_utils as dutil
import util.label_utils as lutil
import util.feature_utils as futil
import util.sync_utils as sutil
import pickle4reducer
import multiprocessing as mp
ctx = mp.get_context()
ctx.reducer = pickle4reducer.Pickle4Reducer()


# In[ ]:


"""
Class that brings together feature and label side to provide all data needed for classification.
Synchronizes the data.
This class is not designed to be used elsewhere. Data should be generated and analysed here, then saved to file using the DataUtil class.
"""
class DataProvider:
    """
    Init function. Start and end time are set for a time period I checked manually to be more or less okay.
    Creates classes to hold label and feature data in memory and classes to create feats and labels.
    Input: draw bool, if we want to visualize the happy/non-happy ratio et al.
    """
    def __init__(self, draw = False):
        self.is_loaded = False #bool to check whether the raw data has already been loaded into memory
        self.draw = draw
        self.all_days_df = None
        
    def _load_raws(self, patient, days): #TODO auf mehrere tage erweitern
        if type(days) is int:
            days = [days]
        #this dataframe saves pat,day,st,end,the raw, non-standardized, non-PCA features and corresponding labels
        all_days_df = pd.DataFrame(columns = ['Patient','Day','Start','End','BinnedData','BinnedLabels', 'GoodChans'], index=(0,len(days)-1))
#         manager = mp.Manager()
#         return_dict = manager.dict()
#         jobs = []
#         with mp.Pool(7) as pool:
#             yass = pool.starmap(self.load_raws_single_day,list(zip([patient]*len(days),days)))
        for enum,day in enumerate(days):
            print(day,'this day')
#             job = mp.Process(target=self.load_raws_single_day, args=(patient,day,return_dict))
#             jobs.append(job)
#             job.start()
            curr_ret = self.load_raws_single_day(patient,day)
            all_days_df.loc[enum] = curr_ret
        #self.annot_data = LabelDataHolder(path_vid,video_start,video_start+self.duration, col = 'annotated')
#         for proc in jobs:
#             proc.join()
#         all_days_df = (pd.DataFrame(return_dict.values(), columns = ['Patient','Day','Start','End','BinnedData','BinnedLabels', 'GoodChans']).sort_values(['Day'])).reset_index(drop=True)
#         all_days_df = (pd.DataFrame(yass, columns = ['Patient','Day','Start','End','BinnedData','BinnedLabels', 'GoodChans']).sort_values(['Day'])).reset_index(drop=True)
        all_days_df = (all_days_df.sort_values(['Day'])).reset_index(drop=True)
        self.all_days_df = all_days_df
        self.featuregen = Feature_generator(all_days_df)
        self.lablegen = Label_generator(all_days_df)
        #das hier erstmal nicht. spaeter zur analyse vielleicht wieder
        #self.annotsgen = Label_generator(all_days_df,mask=self.featuregen.bad_indices['NaNs']) #this is for conf mat later
        self.is_loaded = True
        
    
    def load_raws_single_day(self, patient,day):
        path_ecog, path_vid = sutil.find_paths(patient,day)
        realtime_start, realtime_end = sutil.find_start_and_end_time(path_vid) #output in secs from midnight
        feat_data = FeatDataHolder(path_ecog,realtime_start, realtime_end)
        label_data = LabelDataHolder(path_vid,realtime_start,realtime_end, col = 'Happy_predicted' )
        return [patient, day,realtime_start,realtime_end,feat_data.get_bin_data(),label_data.get_pred_bin(), feat_data.chan_labels]


    def reload_generators(self):
        self.featuregen = Feature_generator(self.all_days_df)
        self.lablegen = Label_generator(self.all_days_df)
        


    """
    Function to generate the feats and labels, given the input hyperparas
    Input: Configs, i.e. Windowsize, sliding window, start and end (in s), train bool, variance to be explained, cutoff if classification.
    Output: Features, Labels
    """
    def generate_data(self,configs, train=True):
        #train data
        if 'expvar' not in configs.keys():
            configs['expvar']=95
        #check whether train or test data, set start and end sample accordingly
        if train:
            start=configs['s_sample']
            end=configs['e_sample']
        else:
            start=configs['s_sample_ev']
            end=configs['e_sample_ev']
        x = self.featuregen.generate_features(wsize = configs['wsize'], start=start,end=end,expl_variance=configs['expvar'],train=train,sliding_window=configs['sliding'])
        y,rat = self.lablegen.generate_labels(wsize = configs['wsize'], start=start,end=end, sliding_window=configs['sliding'])
        #annots, _ = self.annotsgen.generate_labels(wsize=configs['wsize'], start=start,end=end, sliding_window=configs['sliding'])
#         if self.draw:
#             LabelVis.plot_happy_ratio(y,rat)
#             LabelVis.plot_happy_ratio(annots,_)
#             preds = y[~np.isnan(y)] ### this is for the confusion matrix between human annotations and openface labels
#             annots = annots[~np.isnan(y)]
#             preds = preds[~np.isnan(annots)]
#             annots = annots[~np.isnan(annots)] ###
#             ClassificationVis.conf_mat(preds, annots)
        x_clean, y_clean = sutil.filter_data(x,y,self.featuregen.bad_indices)
        return x_clean,y_clean

    
    def get_data(self, configs):
        #if data already exists, simply reload
        try:
            x,y,x_ev, y_ev = dutil.load_data_from_file(configs)
            print('Loading Data from File..done')
        except FileNotFoundError: #file doesn't exist
            print('Data not on file yet. Loading raw data into memory...')
            if not self.is_loaded:
                self._load_raws(configs['patient'],configs['days'])
            print('And creating the data..')
            x,y = self.generate_data(configs, train = True)
            x_ev, y_ev = self.generate_data(configs, train = False)
            print('Done. Saving to file for later use.')
            dutil.save_data_to_file(x,y,x_ev, y_ev, configs)
        #now do the cutoff
        if 'cutoff' in configs.keys(): 
            cutoff = configs['cutoff']
            print('Doing cutoff')
            y = lutil.do_cutoff(y, cutoff)
            y_ev = lutil.do_cutoff(y_ev, cutoff)
        return x,y,x_ev,y_ev
        


# In[ ]:


# provider = DataProvider()


# In[ ]:


# #generate data here, to parallelize
# patient = 'cb46fd46'
# days = [3,4,5]
# wsize = 100
# sliding = 25
# s_sample = 0
# e_sample = 170000
# s_sample_ev = 170000
# e_sample_ev = 200000
# expvar = 90
# configs =dict()
# configs['patient']=patient
# configs['days']=days
# configs['wsize']=wsize
# configs['sliding']=sliding
# configs['s_sample']=s_sample
# configs['e_sample']=e_sample
# configs['s_sample_ev']=s_sample_ev
# configs['e_sample_ev']=e_sample_ev
# configs['expvar'] = expvar


# In[ ]:


#this should not be here, but since stuff isn't doing too well do anyways
# provider._load_raws(configs['patient'], configs['days'])


# In[ ]:


print('los')
#provider.reload_generators()
# x,y,x_ev,y_ev = provider.get_data(configs)


# In[ ]:


# x_ev.shape, y_ev.shape


# In[ ]:


# h,m,h_ev,m_ev = provider.get_data(configs)


# In[ ]:


# m_ev.shape


# In[ ]:


# days=[3,4]
# patient='cb46fd46'
# all_days_df = pd.DataFrame(columns = ['Patient','Day','Start','End','BinnedData','BinnedLabels', 'GoodChans'], index=(0,len(days)-1))
# for enum,day in enumerate(days):
#     print('and go')
#     path_ecog, path_vid = sutil.find_paths(patient,day)
#     realtime_start, realtime_end = sutil.find_start_and_end_time(path_vid) #output in secs from midnight
#     print(realtime_start, realtime_end)
#     feat_data = FeatDataHolder(path_ecog,start=realtime_start, end=realtime_end)
#     label_data = LabelDataHolder(path_vid,0,realtime_end-realtime_start+2000, col = 'Happy_predicted' )
#     all_days_df.loc[enum] = [patient, day,realtime_start,realtime_end,feat_data.get_bin_data(),label_data.get_pred_bin(), feat_data.chan_labels]



# In[ ]:


# featuregena = Feature_generator(all_days_df)
# lablegena = Label_generator(all_days_df)
# print('get x')
# x = featuregena.generate_features(wsize = 100, start=0,end=50000,expl_variance=90,train=True,sliding_window=10)
# print('get y')
# y,rat = lablegena.generate_labels(wsize = 100, start=0,end=50000, sliding_window=10)


# In[ ]:


# #apply the filtering from before
# artis = futil.detect_artifacts(bla)
# bla_clean = futil.remove_artifacts(bla,artis)


# print(bla_clean.shape)
# print(bla.shape)

# #bla_stand = futil.standardize(bla_clean,np.std(bla_clean,axis=1), np.mean(bla_clean,axis=1))
# bla_stand = bla_clean
# high_freq_bins = bla_stand[7::8,:]

# #here, we do the visual analysis of the different bins
# #For the 120 frequency bin, plot distribution of psd over time, over channels
# plt.figure(figsize=(10,10))
# sns.distplot(high_freq_bins.flatten(), kde = False, rug=True, rug_kws={'alpha':.2,'color':'gray'}, hist_kws={'alpha':1})
# plt.title('Histogram of High-Freq bin across channels and time')
# plt.xlabel('PSD')
# plt.ylabel('Counts')


# #now, do the violin plots to check for bad channels
# #first, we create a dictionary out of the individual channels
# dff = pd.DataFrame(columns=['ChanLabel','PSD Values'])
# for enum,chan in enumerate(holder.chan_labels):
#     dff.loc[enum]=(chan,high_freq_bins[enum])
# dff = dff.explode('PSD Values').reset_index(drop=True)
# dff['PSD Values'] = dff['PSD Values'].astype('float')

# plt.figure(figsize=(25,10))
# sns.violinplot(x=dff['ChanLabel'],y=dff['PSD Values'])
# plt.xticks(rotation=45)


# In[ ]:


# import multiprocessing as mp

# def worker(arg1,arg2):
#     arg2[arg1]=[arg1,'lel']


# manager = mp.Manager()
# return_dict = manager.dict()
# jobs = []
# for i in range(3,6):
#     p = mp.Process(target=worker, args=(i,return_dict))
#     jobs.append(p)
#     p.start()


# In[ ]:


# #generate data here, to parallelize
# wsize = 5
# s_sample = 0
# e_sample = 30000
# s_sample_ev = 30000
# e_sample_ev = 35000
# expvar = 90
# configs =dict()
# configs['wsize']=wsize
# configs['s_sample']=s_sample
# configs['e_sample']=e_sample
# configs['s_sample_ev']=s_sample_ev
# configs['e_sample_ev']=e_sample_ev
# configs['expvar'] = expvar


# In[ ]:


# for wsize in [10,30,50,100]:
#     configs['wsize'] = wsize
#     if wsize == 10:
#         for sliding in [False, 3, 5]:
#             configs['sliding'] = sliding
#             x,y, x_ev, y_ev = data.get_data(configs)
#     if wsize == 30: 
#         for sliding in [False,5,15]:
#             configs['sliding'] = sliding
#             x,y, x_ev, y_ev = data.get_data(configs)
#     if wsize == 50: 
#         for sliding in [False,15,25,35]:
#             configs['sliding'] = sliding
#             x,y, x_ev, y_ev = data.get_data(configs)
#     if wsize == 100:
#         for sliding in [False,25,50,75]:
#             configs['sliding'] = sliding
#             x,y, x_ev, y_ev = data.get_data(configs)
        


# In[ ]:


# import matplotlib.pyplot as plt

# x.shape

# np.median(abs(x))

# plt.plot(x[:,0])

# dutil.save_data_to_file(x,y,x_ev,y_ev,configs)


# In[ ]:


# #this is to find cutoff

# bla = DataProvider(draw=True)
# ma = DataProvider(draw=True, col ='annotated')

# y = bla.get_data(sliding=10)
# my = ma.get_data(sliding=10)

# thresh = .3
# for thresh in [.2]:
#     y_class = y.copy()
#     my_class = my.copy()
#     y_class_nans = np.isnan(y_class)
#     my_class_nans = np.isnan(my_class)
#     print(y_class_nans)
#     y_class[y_class<thresh]=0
#     y_class[y_class>0]=1
#     y_class[y_class_nans]=np.nan
#     my_class[my_class<thresh]=0
#     my_class[my_class>0]=1
#     my_class[my_class_nans]=np.nan

#     LabelVis.plot_nan_ratio(my_class)
#     LabelVis.plot_nan_ratio(y_class)

#     ynonan =y_class[~my_class_nans]
#     mynonan = my_class[~my_class_nans]
#     mynonan = mynonan[~np.isnan(ynonan)]
#     ynonan = ynonan[~np.isnan(ynonan)]

#     ClassificationVis.conf_mat(mynonan,ynonan)

