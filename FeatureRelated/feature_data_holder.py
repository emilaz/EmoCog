
# coding: utf-8

# In[1]:


import h5py
from .simple_edf_preprocessing import Preprocessor
import numpy as np


# In[ ]:


class FeatDataHolder:
    
    def __init__(self,path):
        #sampling frequency and last sample taken
        df=h5py.File(path)
        self.sfreq=int(df['f_sample'][()])
        #TODO THIS IS ENTERED MANUALLY FOR DAY 4, PAT c46fd46!!! CHANGE TO READ FROM FILE ONCE AVAILBABLE
        ###so the actual portion of the data we want is this, in s
        print('Warning. Start and end points for the given dataset is still manually set for patient c46fd46, day 4.')
        self.start=11
        self.end=43205
        #preprocess data
        preprocessor=Preprocessor(df,start_sample=int(self.start*self.sfreq),end_sample=int(self.end*self.sfreq))
        self.data,self.bad_chan,self.bad_idx=preprocessor.preprocess(prefiltered_sd_kurt=True)
        self.data=self.data[self.bad_chan!=True]
        self.chan_labels=np.array(eval(df['chanLabels'][()]))[self.bad_chan!=True]
        #how many samples in this dataset?
        self.data_bin,self.mask_bin=self._bin_data()
        

    #this function restructures the data into a 3D structure, where each row presents a channel, each column one second
    #and the depth is the amount of samples per seconds (sfreq). 
    #This is to discard seconds where bad_idx are present and to be on par with the labels in the end
    #This function also creates a mask of bins to discard from the bad_idx array
    def _bin_data(self):
        #where to end?
        data_bin=self.data.reshape(self.data.shape[0],self.end-self.start,self.sfreq)
        mask_bin=np.all(self.bad_idx.reshape(self.end-self.start,self.sfreq),axis=1)
        return data_bin, mask_bin
    
    def get_bin_data_and_mask(self):
        return self.data_bin,self.mask_bin


