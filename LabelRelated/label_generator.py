import numpy as np
import dask
import pandas as pd

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

    # def generate_labels(self, wsize=100, sliding_window=0, method='ratio'):
    #     curr_data = None
    #     ratio = None
    #     for day in self.df['Day']:
    #         print('Day', day)
    #         data = self.df[self.df['Day'] == day].BinnedLabels.values[0]
    #         mat, rat = self._generate_labels_single_day(data, wsize, sliding_window)
    #         if curr_data is None:
    #             curr_data = mat
    #             ratio = rat
    #         else:
    #             curr_data = np.append(curr_data, mat, axis=None)
    #             ratio = np.append(ratio, rat, axis=None)
    #     return curr_data, ratio

    def generate_labels(self, wsize=100, sliding_window=False, method='ratio'):
        results = []
        for idx, row in self.df.iterrows():
            data = row['BinnedLabels']
            res = dask.delayed(self._generate_labels_single_day)(data, wsize, sliding_window)
            # res[0] is the data, res[1] are the bad indices
            results.append([row['Patient'], row['Day'], res[0], res[1]])
        res = dask.compute(*results)
        df = pd.DataFrame(res,columns=['Patient','Day','Y','Ratio']).sort_values(
            ['Day', 'Patient']).reset_index(drop=True)
        return df
