import sys
import numpy as np
from scipy import signal
import pickle
from multiprocessing import Pool
import pandas as pd
import util.feature_utils as util
import os
import glob

"""
Class to generate features. Uses preprocessed data held in memory by FeatDataHolder class.
"""


class FeatureGenerator:
    """
    Init function.
    """

    def __init__(self, df):
        # sampling frequency and last sample taken
        self.sfreq = 500  # will always be, hopefully lol
        # load raw feat data, filter to common channels, then save to file. This due to big data size
        self.df = util.filter_common_channels(df)

        self.good_channels = self.df['GoodChans'].iloc[0]

    """
    Function needed for calculating the features. Central piece on the feature side. Works as follows:
    columns: channels, for each channel the 150 frequencies (0-150Hz) (hece freq*cha length), binned logarithmically
    Rows: Time steps, defined by sliding window+window size
    resulting matrix is 2D, Time Stepsx(Freq*Channels)
    In case of generating train data, this function also saves mean and stddev for standardization purpose.
    Input:  window size and sliding window in sec
    Output: Standardized, binned data.
    """

    def _generate_features_single_day(self, data, wsize=100, sliding_window=False):
        data = pickle.load(open(data, 'rb'))  # from link to data
        bads = []
        time_it = 0
        mat = None
        idx = 0
        while True:
            stop = time_it + wsize
            if stop > data.shape[1]:
                print('This day gave us {} seconds worth of data'.format(data.shape[1] - 1))
                break
            # Note that each column is exactly one second.
            # get data in range of ALL channels
            curr_data = data[:, time_it:stop, :].reshape(data.shape[0], -1)
            # welch method
            fr, psd = signal.welch(curr_data, self.sfreq, nperseg=250)

            # if there are nans in the psd, something's off. throw away, save index, continue
            if np.isnan(psd).any():
                bads += [idx]  # current index bad
                if sliding_window:
                    time_it += sliding_window
                else:
                    time_it += wsize
                idx += 1
                continue

            fr_bin, psd_bin = util.bin_psd(fr, psd)
            idx += 1
            if mat is None:
                # first time. create first column, flatten w/o argument is row major
                mat = psd_bin.flatten()
            else:
                # after, add column for each time step
                mat = np.column_stack((mat, psd_bin.flatten()))
            # sliding window?
            if sliding_window:
                time_it += sliding_window
            else:
                time_it += wsize
        return mat, bads  # we do the standardization after the filtering

    def generate_features(self, wsize=100, sliding_window=False):
        # save the data to files bc multiprocessing can't handle large files
        data_links = self.df['BinnedData'].values
        pass_me = zip(data_links, [wsize] * len(data_links), [sliding_window] * len(data_links))
        pats =  self.df['Patient']
        days = self.df['Day']
        p = Pool(8)
        res = p.starmap(self._generate_features_single_day, pass_me)
        df = pd.DataFrame(data=res, columns=['X', 'BadIndices'])
        df['Patient'] = pats  # we use the fact that map() returns are ordered
        df['Day'] = days
        df = df.sort_values(
            ['Day','Patient']).reset_index(drop=True)
        return df

    """
    Function to return the bad indices found by filtering. 
    Important: First filter out the nan indices, then the artifacts!
    Order is important
    Output: Dictionary of bad data points.
    """

    def get_bad_indices(self):
        raise NotImplementedError(
            'This function uses deprecated data structures and should not be used.')

