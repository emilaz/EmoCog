import sys
import numpy as np
from scipy import signal
import dask
import pandas as pd
import util.feature_utils as util
# sys.path.append('..')


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
        self.df = util.filter_common_channels(df)
        # this needs to be passed on to the label side. Will include bad indices found during calculation
        self.good_channels = self.df['GoodChans'].loc[0]

    """
    Function needed for calculating the features. Central piece on the feature side. Works as follows:
    columns: channels, for each channel the 150 frequencies (0-150Hz) (hece freq*cha length), binned logarithmically
    Rows: Time steps, defined by sliding window+window size
    resulting matrix is 2D, Time Stepsx(Freq*Channels)
    In case of generating train data, this function also saves mean and stddev for standardization purpose.
    Input:  Start and end time (in secs), 
            bool for whether train data or not (for PCA), 
            window size and sliding window in sec
    Output: Standardized, binned data.
    """

    def _generate_features_single_day(self, data, wsize=100, sliding_window=False):
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
                bads += [idx]  # current index baad
                if (sliding_window):
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

    # def generate_features(self, wsize = 100, sliding_window=False):
    #     # here, check how many days we need for the requested datasize
    #     curr_data = None
    #     for day in self.df['Day']:
    #         print('Day', day)
    #         data = self.df[self.df['Day'] == day].BinnedData.values[0]
    #         mat, bad = self._generate_features_single_day(data, wsize, sliding_window)
    #         if curr_data is None:
    #             curr_data = mat
    #             self.bad_indices = np.array(bad)
    #         else:
    #             self.bad_indices = np.append(self.bad_indices,np.array(bad)+len(self.bad_indices)+curr_data.shape[1])
    #             if mat is not None:
    #                 curr_data = np.append(curr_data,mat,axis=1)
    #     return curr_data

    def generate_features(self, wsize=100, sliding_window=False):
        results = []
        for idx, row in self.df.iterrows():
            data = row['BinnedData']
            res = dask.delayed(self._generate_features_single_day)(data, wsize, sliding_window)
            # res[0] is the data, res[1] are the bad indices
            results.append([row['Patient'], row['Day'], res[0], res[1]])
        res = dask.compute(*results)
        df = pd.DataFrame(data=res, columns=['Patient', 'Day', 'X', 'BadIndices']).sort_values(
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

# def filter_common_channels(common_df):
#     good_idx = util.find_common_channels(common_df[['Day','GoodChans']])
#     for idx,day in enumerate(good_idx['Day']):
#         good = good_idx.loc[good_idx['Day']==day,'CommonChans'][idx]
#         new_data = common_df[common_df['Day']==day]['BinnedData'][idx][good,:]
#         common_df.loc[common_df['Day']==day,'BinnedData'] = [new_data]
#         spraa = common_df.loc[common_df['Day']==day,'GoodChans'][idx][good]
#         common_df.loc[common_df['Day']==day,'GoodChans'] = [spraa]
#     return common_df
