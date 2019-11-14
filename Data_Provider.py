#!/usr/bin/env python
# coding: utf-8

# In[1]:


from FeatureRelated.feature_generator import Feature_generator
from LabelRelated.label_generator import Label_generator 
from FeatureRelated.feature_data_holder import FeatDataHolder
from LabelRelated.label_data_holder import LabelDataHolder
import numpy as np
from Vis import LabelVis, ClassificationVis
from Util import DataUtils as dutil
from Util import LabelUtils as lutil


# In[2]:


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
        
    def _load_raws(self):
        realtime_start = 29413 #this is 8h, 10min and 13s into day 4
        video_start = realtime_start - 29344 # 29344 is the beginning of recordings of video data (see /home/emil/data/sync_data)
        self.duration = 37820+4500 #in seconds, of course
        self.feat_data = FeatDataHolder('/data2/users/stepeter/Preprocessing/processed_cb46fd46_4.h5',start=realtime_start, duration=self.duration)
        self.label_data = LabelDataHolder('/home/emil/data/hdf_data/cb46fd46_7_imp_columns.hdf',video_start,video_start+self.duration, col = 'Happy_predicted' )
        self.annot_data = LabelDataHolder('/home/emil/data/hdf_data/cb46fd46_7_imp_columns.hdf',video_start,video_start+self.duration, col = 'annotated')
        self.featuregen = Feature_generator(self.feat_data)
        self.lablegen = Label_generator(self.label_data,mask=self.featuregen.mask_bin)
        self.annotsgen = Label_generator(self.annot_data,mask=self.featuregen.mask_bin) #this is for conf mat later

    #get datasets
#     """
#     Function to generate the feats and labels, given the input hyperparas
#     Input: Windowsize, sliding window, start and end (in s), train bool, variance to be explained, cutoff if classification.
#     Output: Features, Labels
#     """
#     def generate_data(self,configs, train=True, expl_var=95, cutoff = None):
#         if e_sample == None:
#             e_sample = self.duration-1
#         #train data
#         x = self.featuregen.generate_features(wsize = wsize, start=s_sample,end=e_sample,expl_variance=expl_var,train=train,sliding_window=sliding)
#         y,rat = self.lablegen.generate_labels(wsize = wsize, start=s_sample,end=e_sample, sliding_window=sliding, cutoff=cutoff)
#         annots, _ = self.annotsgen.generate_labels(wsize=wsize, start=s_sample,end=e_sample, sliding_window=sliding, cutoff=cutoff)
#         if self.draw:
#             LabelVis.plot_happy_ratio(y,rat)
#             LabelVis.plot_happy_ratio(annots,_)
#             preds = y[~np.isnan(y)] ### this is for the confusion matrix between human annotations and openface labels
#             annots = annots[~np.isnan(y)]
#             preds = preds[~np.isnan(annots)]
#             annots = annots[~np.isnan(annots)] ###
#             ClassificationVis.conf_mat(preds, annots)
#         x = x[~np.isnan(y)]
#         y = y[~np.isnan(y)]
#         return x,y

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
        annots, _ = self.annotsgen.generate_labels(wsize=configs['wsize'], start=start,end=end, sliding_window=configs['sliding'])
        if self.draw:
            LabelVis.plot_happy_ratio(y,rat)
            LabelVis.plot_happy_ratio(annots,_)
            preds = y[~np.isnan(y)] ### this is for the confusion matrix between human annotations and openface labels
            annots = annots[~np.isnan(y)]
            preds = preds[~np.isnan(annots)]
            annots = annots[~np.isnan(annots)] ###
            ClassificationVis.conf_mat(preds, annots)
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        return x,y

    def get_data(self, configs):
        #if data already exists, simply reload
        try:
            x,y,x_ev, y_ev = dutil.load_data_from_file(configs)
            print('Loading Data from File..done')
        except FileNotFoundError: #file doesn't exist
            print('Data not on file yet. Loading raw data into memory...')
            if not self.is_loaded:
                self._load_raws()
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
        


# In[3]:


# data = DataProvider(draw=True)


# In[4]:


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


# In[5]:


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

