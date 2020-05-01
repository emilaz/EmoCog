import numpy as np
import pickle
import os
import glob
import pandas as pd
from multiprocessing import Pool


"""
This class is used to generate usable labels from the data preprocessed by the LabelDataHolder class. 
Class was initially created to avoid reloading data into memory every time functions were altered/debugged.
"""


class LabelGenerator:
    """
    Init function.
    Input: DF with label data of (possibly) several days
    """

    def __init__(self, df):
        self.fps = 30
        # df['BinnedLabels'] = [pickle.load(open(f, 'rb')) for f in df['BinnedLabels']]
        self.df = df

    """
    Generates labels. Use sliding window if features are also generated with sliding window. 
    Note that we specified a time frame in the LabelDataHolder class.
    Within that frame, we can specify start and end point for the labels generated.
    Useful for splitting into train/test data.
    Input:  Start point, 
            End Point (in seconds), 
            Windowsize for subsuming, 
            method for subsuming labels in given window (ratio or median),
            Cutoff if we need boolean labels).
    Output: Usable labels for regression/classification task. 
            If classification is used, also returns the percentage of non-nan labels per label generated.
    """

    def _generate_labels_single_day(self, data, wsize=100, sliding_window=0, method='ratio'):
        if method != 'ratio' and method != 'median':
            raise NameError('The given method does not exist. Try one of the following: ratio,median.')
        if method is 'median':
            print('Note: The median method is currently a 75 percentile.')

        data = pickle.load(open(data, 'rb'))  # from link to data
        # average "happiness" per second
        happy_portion = np.nanmean(np.array(data, dtype='float'), axis=1)
        non_nans_per_s = np.count_nonzero(~np.isnan(np.array(data, dtype='float')), axis=1)
        labels = []
        good_ratio = []
        time_it = 0
        while True:
            stop = time_it + wsize
            if stop > data.shape[0]:
                print('This day gave us {} secs worth of data'.format( data.shape[0] - 1))
                break
            curr_data = happy_portion[time_it:stop]
            curr_non_nans = np.sum(non_nans_per_s[time_it:stop])
            if not curr_data.size:
                print('SOMETHINGS WRONG HERE. CHECK.')
                if sliding_window:
                    time_it += sliding_window
                else:
                    time_it += wsize
                continue
            # here, we divide by len(curr_data), because
            # we don't want the influence of nans that were thrown away
            # due to bad feature points.
            good_ratio += [float(curr_non_nans) / (self.fps * len(curr_data))]
            if method == 'ratio':
                labels += [np.nanmean(curr_data)]
            elif method == 'median':
                labels += [np.nanpercentile(curr_data, q=75)]
            if sliding_window:
                time_it += sliding_window
            else:
                time_it += wsize
        labels = np.array(labels)
        return labels, good_ratio  # THIS IS NOT USED SO FAR, BUT SHOULD BE.

    def generate_labels(self, wsize=100, sliding_window=False, method='ratio'):
        data_links = self.df['BinnedLabels'].values
        pass_me = zip(data_links,[wsize]*len(data_links), [sliding_window]*len(data_links))
        pats = self.df['Patient']
        days = self.df['Day']
        p = Pool(8)
        res = p.starmap(self._generate_labels_single_day, pass_me)
        df = pd.DataFrame(res, columns=['Y','Ratio'])
        df['Patient'] = pats  # we use the fact that map() returns are ordered
        df['Day'] = days
        df = df.sort_values(
            ['Day', 'Patient']).reset_index(drop=True)
        return df
