#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#import mne
import matplotlib.pyplot as plt
import h5py


# In[ ]:


"""
Class for preprocessing the ECoG data.
"""

class Preprocessor:
    
    """
    Init function. Takes data as given by FeatureDataHolder class, sets the bad indices and bad channels.
    Input: Data, start and end in sample no. (not! seconds)
    """
    def __init__(self,data=None,chan_info=None,start_sample=0,end_sample=None):
        self.data=data
        self.start=start_sample
        if end_sample is None:
            self.end=data['dataset'].shape[1]-1
        else:
            self.end=end_sample
        self.df=data['dataset'][:,start_sample:end_sample]
        self.bad_chans=~(np.array([chan_info[c]['goodChanInds'] for c in chan_info.columns]).astype('bool'))
        

    """
    Function to call both functions for preprocessing, one for channels, one for indices (see above)
    Input: Which of the two we want
    Output: The dataframe itself, in given time window (start and end time), indices of bad chans and bad indices
    """
    def preprocess(self,prefiltered_sd_kurt=True,by_artifact=True):
        #self.__flat_sig()
        #self.__bad_chan()
        return self.df, self.bad_chans
    


# In[ ]:


# df=h5py.File('/data2/users/stepeter/Preprocessing/processed_a0f66459_4.h5')
# pr=Preprocessor(df,start_sample=11*500,end_sample=43205*500)


# In[ ]:


# dat,tt,tw=pr.preprocess()


# In[ ]:


# print(dat.shape)
# print(tt.shape)
# tw.shape


# In[ ]:


# print(np.unique(tw,return_counts=True))


# In[ ]:


# print(tt.shape)
# print(tw.shape)
# print(dat.shape)
# aight=dat[tt!=True]
# aight=aight[:,tw!=True]
# print(aight.shape)
# print(tw[-1:])


# In[ ]:


# prepr.bad_var.shape#print(np.unique(prepr.bad_idx,coun))

# print(prepr.data[0].shape)
# print(prepr.bad_idx.shape)
# for p in range(0,127):
#     if not prepr.bad_chans[p]: 
#         if not prepr.bad_var[p]:
#             plt.plot(prepr.data[p,prepr.bad_idx[p]!=True])
#     #mark bad points
#     #plt.plot(np.arange(len(prepr.bad_idx[p]))[prepr.bad_idx[p]],np.zeros(len(prepr.bad_idx[p]))[prepr.bad_idx[p]]-.02, "-", markersize=1,)
# plt.legend()

# for p in range(0,127):
#     if (bad_chans)
#     plt.plot(prepr.data[p])

# plt.plot(tt[0])
# plt.plot(tt[1])
# plt.plot(peaks, tt[1][peaks], "x")


# np.random.seed(33)
# test=np.random.uniform(0.9,1,(1,100))
# print(test.shape)
# test[0,10:20]=0
# #test[1,15:25]=0
# test[0,40:73]=0.95
# #test[1,15:25]=0.95
# #peaks=argrelextrema(test,np.less,order=25)
# #plt.vlines(peaks,ymin=0, ymax=1)
# #print(peaks)
# grads=(np.gradient(test,axis=1)==0)
# print(np.zeros(len(grads[p]))[grads[p]].shape)
# for p in range(test.shape[0]):
#     plt.plot(test[p])
#     ym1 = np.ma.masked_where(grads[p] ==False , grads[p])
#     plt.plot(np.arange(len(grads[p])),ym1-1.02, "-")

# #np.argwhere(grads[1])[0]
# #print(grads)
# np.apply_along_axis(np.argwhere,1,grads)
# np.argwhere(grads[2])
# np.apply_along_axis(np.unique,1,grads)


# ee=np.eye(3)
# mask=np.zeros(ee.shape[0],dtype='bool')
# mask[1]=True
# mask[0]=True
# #print(mask)

# test=np.zeros(5)
# np.logical_or(np.ones(5),test==1)


# A=np.arange(5)
# A[np.arange(2,3)]=False
# print(A)

