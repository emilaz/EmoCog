import sys

sys.path.append('..')
import h5py
import numpy as np
import pandas as pd

"""
Class for holding feature data in memory. Also invokes preprocessing of the data
"""


class FeatDataHolder:
    """
    Init class. Loads data in given time window, invokes preprocessor.
    Input: Path to data, start and end (in s).
    """

    def __init__(self, path, start=0, end=None):
        # sampling frequency and last sample taken
        df = h5py.File(path)
        chan_info = pd.read_hdf(path, key='chan_info')
        self.sfreq = int(df['f_sample'][()])
        self.start = start
        self.end = end
        start_sample = round(self.start * self.sfreq)
        end_sample = round(self.end * self.sfreq)
        self.data = df['dataset'][:, start_sample:end_sample]
        self.bad_chan = ~(np.array([chan_info[c]['goodChanInds'] for c in chan_info.columns]).astype('bool'))
        self.data = self.data[self.bad_chan != True]
        self.chan_labels = np.array([c for c in chan_info.columns])[self.bad_chan != True]
        if len(self.data.flatten()) % 500 != 0:
            raise ValueError('The features data here does not have the correct shape for some reason. '
                             'Shape: {}, path {}'.format(self.data.shape, path))
        self.data_bin = self._bin_data()

    """
    This function restructures the data into a 3D structure.
    Each row presents a channel, each column one second and the depth is the amount of samples per seconds (sfreq). 
    This is to discard seconds where bad_idx are present and to be on par with the labels in the end
    This function also creates a mask of the bins to later discard from the bad_idx array
    Output: Binned data, corresponding mask
    """

    def _bin_data(self):
        # where to end?
        data_bin = self.data.reshape(self.data.shape[0], -1, self.sfreq)
        return data_bin

    """
    Returns the data and mask bin
    Output: Binned data, corresponding mask
    """

    def get_bin_data(self):
        return self.data_bin

    """
    Sets bad indices manually
    Input: Start and end of bad indices
    """

    def set_bad_idx(self, idx_start, idx_end):
        self.bad_idx[idx_start:idx_end] = True

    """
    Sets bad chan manually
    Input: Chan to be discarded
    """

    def set_bad_chan(self, chan):
        self.data = self.data[np.arange(self.data.shape[0]) != chan]
