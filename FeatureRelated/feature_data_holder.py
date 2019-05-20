
# coding: utf-8

# In[6]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import sys
sys.path.append('..')
import h5py
from .simple_edf_preprocessing import Preprocessor
#from simple_edf_preprocessing import Preprocessor
from Vis import FeatureVis
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[7]:


class FeatDataHolder:
    
    def __init__(self,path,start=0,duration=0):
        #sampling frequency and last sample taken
        df=h5py.File(path)
        self.sfreq=int(df['f_sample'][()])
        #TODO THIS IS ENTERED MANUALLY FOR DAY 4, PAT c46fd46!!! CHANGE TO READ FROM FILE ONCE AVAILBABLE
        #day 4 video recodings start at 08:10:12:844 and go uninterrupted for 12 huors (check in data/sync_data)
        #start date in seconds from 12AM is 29.413
        ###so the actual portion of the data we want is this, in s
        print('Warning. Start and end points for the given dataset is still manually set for patient c46fd46, day 4.')
        #both in s
        self.start = start
        #self.end=72608 #THIS IS UNTIL VIDEO RECORDINGS ARE INTERRUPTED
        self.end = self.start + duration #this is till where i explored things, around 10h of recordings amk
        #preprocess data
        preprocessor = Preprocessor(df,start_sample=int(self.start*self.sfreq),end_sample=int(self.end*self.sfreq))
        self.data,self.bad_chan,self.bad_idx = preprocessor.preprocess(prefiltered_sd_kurt=True)
        self.data = self.data[self.bad_chan!=True]
        self.chan_labels = np.array(eval(df['chanLabels'][()]))[self.bad_chan!=True]
        #how many samples in this dataset?
        print('Warning. Some datapoints are manually set to be bad. This only holds true for pat cb46fd46, day 4.')
        #excluding last electrode
        self.set_bad_chan(self.data.shape[0]-1)
        #excluding some points found by hand lol, this better be right WARNING: THIS IS RELATIVE TO SET STARTING TIME
        bad_indices = [[0,60000],[1505000,1515000],[2130000,2180000],[3055000,3460000],[4140000,4150000],[2535000,2540000],[4140000,4145000],[14370000,14390000],[14440000,14447000],[14635000,14680000],[18910000+24300,18910000+24600],[18910000+1187000,18910000+1188000],[18910000+1858000,18910000+1861000],[18910000+2010000,18910000+2015000]]
        for idces in bad_indices:
            self.set_bad_idx(idces[0],idces[1])
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
    
    def set_bad_idx(self, idx_start,idx_end):
        self.bad_idx[idx_start:idx_end] = True
        
    def set_bad_chan(self,chan):
        self.data = self.data[np.arange(self.data.shape[0])!=chan]



# In[13]:


# realtime_start = 29413 #this is 8h, 10min and 13s into day 4 PLUS 10 hours
# durr = 37820+7200
# lel = FeatDataHolder('/data2/users/stepeter/Preprocessing/processed_cb46fd46_4.h5',start=realtime_start, duration = durr)


# lel.data.shape
# FeatureVis.plot_raw_data(lel.data[:,(37820)*500:])#,bad_coords=[[0,60000],[1505000,1515000],[2130000,2180000],[3055000,3460000],[4140000,4150000]])

