#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from Util import LabelUtils as util


# In[ ]:


"""
This class loads label data from file into memory and preprocesses it to be easily usable (binning etc.) for generating labels
"""
class LabelDataHolder:
    """
    Init function. Loads data in given start-end window and bins the data.
    Input: Path to label file, start and end point (in seconds) of labels wanted, type of label wanted (predicted vs. annotated)
    """
    def __init__(self,path,start=0,end=None, col='Happy_predicted'):
        if path.endswith('.csv'):
            self.df=pd.read_csv(path,error_bad_lines=False, low_memory=False)#,dtype={'realtime':'datetime64'})
        elif path.endswith('.hdf'):
            self.df=pd.read_hdf(path,error_bad_lines=False, low_memory=False)#,dtype={'realtime':'datetime64'})
        self.fps=30   #Update June: Apparently openface corrects fr to 30FPS
        self.start=start*self.fps
        if end is None:
            self.end=self.df.shape[0]
        else:
            self.end=end*self.fps
        labels=self.create_label_array(self.start,self.end,col)
        self.pred_bin=self._bin_preds(labels)
        
        
    """
    Given the dataframe and desired number of frames, return array with labels.
    Input: Whole df, number of frames wnated
    Output: Big array
    """
    
    def create_label_array(self,start,end, col):
        vid_nos = self.df['vid'].unique()
        last_vid = vid_nos[-1] #assuming ordering
        big_array = None
        first = True
        fill_vid_after = [False]
        sanity = False #this just for debug
        for vid in vid_nos:
            actual_frames = self.df[self.df['vid']==vid][col].values #actual frames saved in hdf from gautham
            if col == 'annotated': #the annotated labels are strings. Convert here.
                actual_frames = util.convert_labels_readable(actual_frames)
            supposed_no_frames = util.find_number_frames(self.df,vid, last_vid, fill_vid_after) #how many frames does the video actually have?
            ret = util.fill_frames(actual_frames,supposed_no_frames) #fill the frames
            if fill_vid_after[0]: #for some reason, some videos are missing from hdf. Fill in nans for this here
                empty_vid = np.empty(120*self.fps)
                empty_vid[:]=np.nan
                ret = np.append(ret,empty_vid)
                fill_vid_after[0] = False
            if first:
                big_array = ret
                first = False
            else:
                big_array = np.concatenate((big_array,ret),axis=None)
            if len(big_array)>=end:
                big_array[start:end]
                sanity = True
                break
        if not sanity:
            print('Jo hier ist was falsch. Angeblich nicht genug Datan der Bastard?')
        return big_array
        
    """
    Function for binning labels. Also converts the char predictions ('Happy'/'Not Happy') into usable bools if needed.
    Input: Column wanted
    Output: Binnned labels (one row = one sec)
    """
    def _bin_preds(self,labels):
        #bin s.t. each column is one sec.
        end=labels.shape[0]//self.fps
        return labels[:self.fps*end].reshape(-1,self.fps)
    
    def get_pred_bin(self):
        return self.pred_bin

