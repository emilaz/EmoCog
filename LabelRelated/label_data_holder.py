
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[1]:


class LabelDataHolder:
    
    def __init__(self,path,start=0,end=None, col='Happy_predicted'):
        if path.endswith('.csv'):
            self.df=pd.read_csv(path,error_bad_lines=False, low_memory=False)#,dtype={'realtime':'datetime64'})
        elif path.endswith('.hdf'):
            self.df=pd.read_hdf(path,error_bad_lines=False, low_memory=False)#,dtype={'realtime':'datetime64'})
        self.fps=31
        #account for slighlty lower framerate due to openface
        self.start=start*self.fps
        if end is None:
            self.end=self.df.shape[0]
        else:
            self.end=end*self.fps
        self.df=self.df.iloc[self.start:self.end]
        #self._convert_to_unix_time()
        self.pred_bin=self._bin_preds(col)
    
    def _bin_preds(self, col):
        annot=self.df[col].values
        if col == 'annotated':
            nan_indices= annot=='N/A'
            annot[annot!='Happy']=0
            annot[annot=='Happy']=1
            annot[nan_indices]=np.nan
        #bin s.t. each column is one sec.
        end=annot.shape[0]//self.fps
        return annot[:self.fps*end].reshape(-1,self.fps)
    
    def get_pred_bin(self):
        return self.pred_bin

