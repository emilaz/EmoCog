import numpy as np


"""
Class for preprocessing the ECoG data.
"""

class Preprocessor:
    
    """
    Init function. Takes data as given by FeatureDataHolder class, sets the bad indices and bad channels.
    Input: Data, start and end in sample no. (not! seconds)
    """
    def __init__(self,data,chan_info,start_sample,end_sample):
        self.df=data['dataset'][:,start_sample:end_sample]
        self.bad_chans=~(np.array([chan_info[c]['goodChanInds'] for c in chan_info.columns]).astype('bool'))
        

    """
    Function to call both functions for preprocessing, one for channels, one for indices (see above)
    Input: Which of the two we want
    Output: The dataframe itself, in given time window (start and end time), indices of bad chans and bad indices
    """
    def preprocess(self,prefiltered_sd_kurt=True,by_artifact=True):
        return self.df, self.bad_chans