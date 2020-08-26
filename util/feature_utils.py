import numpy as np
import pandas as pd
from functools import reduce
from sklearn.decomposition import PCA
import pickle


def standardize(data, std, data_mean):
    """ Standardizes data (demean& unit variance)
    Input: Data, standart deviation of data, mean of data
    """
    data_dem = data - data_mean[:, None]
    data_stand = data_dem / (std[:, None])
    # print('WARNING. NO STANDARDIZATION AT THE MOMENT. DO YOU WANT THIS?')
    # data_stand = data_dem
    return data_stand


def detect_artifacts(data_matrix, std_lim=None, med_lim=None):
    """ Return indices of bad time points, based on outlier detection
    Input: Data matrix
    Output: Bad indices matrix, same length as data matrix
    """
    # first, we only want to look at the high-freq bin:
    high_freqs = data_matrix[7::8, :]
    # next, calculate median std across each channel(should be 1 I think)
    if std_lim is None:
        std_lim = np.std(high_freqs, axis=1)
        med_lim = np.median(high_freqs, axis=1)
    # next, for each row, get all indices that are too far away from the median in one direction
    too_high = high_freqs > (med_lim + 4 * std_lim)[:, None]  # should yield a 2D matrix
    # for each column, check if there is an entry that's too high
    too_high_idx = np.any(too_high, axis=0)
    # same for other direction
    too_low = high_freqs < (med_lim - 4 * std_lim)[:, None]  # should yield a 2D matrix
    too_low_idx = np.any(too_low, axis=0)
    bad_idx = too_low_idx | too_high_idx  # any index either too high or too low? that is a bad index
    return bad_idx, std_lim, med_lim

def bin_psd(fr, psd):
    """
    Key function for feature processing. Truncates frequencies above 150Hz, bins the frequencies logarithmically.
    Throws the PSD into these bins by summing all PSD that fall into a certain bin.
    Input: Frequency array, PSD array
    Output: Binned frequencies, binned PSD
    """
    fr_trun = fr[fr <= 150]
    fr_total = len(fr_trun)
    fr_bins = np.arange(int(np.log2(max(fr_trun)) + 1))  # just np.arange(no_fr_bins)
    # truncate everythin above 150Hz
    psd = psd[:, fr <= 150]
    psd_bins = np.zeros((psd.shape[0], len(fr_bins)))
    prev = 0
    # these are the general upper limits. they don't give info where in fr/psd these frequencies acutally are!
    max_psd_per_bin = np.exp2(fr_bins).astype('int')
    # hence we need this method. It gives us a 2D array with the number of rows = number of bins,
    # where each row contains 2 values, upper and lower limit of that bin
    prev = 0
    limits = np.zeros((max_psd_per_bin.shape[0], 2), dtype='int')
    for en, b in enumerate(max_psd_per_bin):
        if en == 0:
            arr = np.where((fr_trun >= prev) & (fr_trun <= b))[0]
        else:
            arr = np.where((fr_trun > prev) & (fr_trun <= b))[0]
        check = np.array([min(arr), max(arr)])
        limits[np.log2(b).astype('int')] = check
        prev = b
    prev = 0
    # now, we will the psd_bins by adding up all psd in the range defined before. The last bin gets all remaining frequencies
    # meaning, if the last bin ranges from 64-128, it'll also include everything until 150.
    for b in fr_bins:
        if (b == fr_bins[-1] or limits[b][1] >= fr_total):
            psd_bins[:, b] += np.sum(psd[:, limits[b, 0]:], axis=1)
        else:
            psd_bins[:, b] = np.sum(psd[:, limits[b, 0]:limits[b, 1] + 1], axis=1)
    return fr_bins, psd_bins





def get_no_comps(data, expl_var_lim):
    """Function for PCA.
    Given some minimum of explained variance of the data, return the number of components needed (at most 100).
    Input: Data, desired variance explained
    Output: Number of Components needed.
    """
    comps = min(100, min(data.shape))
    pca = PCA(n_components=comps)
    pca.fit(data)
    tot = 0
    for idx, c in enumerate(pca.explained_variance_ratio_):
        tot += c
        if tot * 100 > expl_var_lim:
            return idx + 1
    return pca.n_components_


def find_common_channels(channels_per_day, additional_channels):
    """ Function to find the indices of channels that were good across days.
    Input: PD Series with good channel info per day
    Output: Good Indices?
    """
    common = reduce(np.intersect1d, channels_per_day)  # which channels are good on all days?
    if additional_channels is not None:
        common = reduce(np.intersect1d, np.array((common, additional_channels)))
    return common


def find_common_channel_indices(channels_per_day, common_channels):
    # get the indices of these common channels, for each day
    good_df = pd.DataFrame(columns=['Patient', 'Day', 'CommonChans'])
    for idx, row in channels_per_day.iterrows():
        indices = [np.min(np.nonzero(row['GoodChans'] == ch)[0]) for ch in
                   common_channels]  # these are the indices to keep - not necessarily in correct order!
        good = np.zeros(len(row['GoodChans']), dtype='bool')  # therefore, bool array for order
        good[indices] = True
        good_df.loc[idx] = [row['Patient'], row['Day'], good]
    return good_df


def filter_common_channels(common_df, additional_channels=None):
    """ This function filters out bad channels across days. For test data, an additional_channels parm is provided
    :param common_df: the dataframe containing patient, day, goodchans, data columns
    :param additional_channels: potential additional channels that are used for intersectional filtering
    :return: a dataframe with filtered channels and data (or rather, links to said data)
    """
    good_common_chans = find_common_channels(common_df['GoodChans'], additional_channels)
    good_idx_df = find_common_channel_indices(common_df[['Patient', 'Day', 'GoodChans']], good_common_chans)
    new_chan_col = []  # don't modify a df you're iterating over (as per docs)
    new_df_col = []
    for idx, row in good_idx_df.iterrows():
        pat = row['Patient']
        day = row['Day']
        good = good_idx_df[(good_idx_df['Patient'] == pat) &
                           (good_idx_df['Day'] == day)]['CommonChans'].iloc[0]  # True/False array

        data_link = common_df[(common_df['Patient'] == pat) & (common_df['Day'] == day)]['BinnedData'].iloc[0]
        new_data = pickle.load(open(data_link,'rb'))
        new_data = new_data[good, :, :]
        filtered_chans = common_df.loc[(common_df['Patient'] == pat) &
                                       (common_df['Day'] == day)]['GoodChans'].iloc[0][good]  # channels
        # sort the order of electrodes to be consistent
        if idx == 0:
            first_order = {k: v for v, k in enumerate(filtered_chans)}  # these are chan-index pairs
        else:
            new_order = [first_order[ent] for ent in filtered_chans]  # these are indices
            # just a safety measure here
            if (filtered_chans != filtered_chans[new_order]).any():
                print('Order of channels seem to be different across patients/days? Check!')
                print(filtered_chans, 'current')
                print(new_chan_col[0], 'reference')
            if (new_order != np.arange(len(new_order))).any():
                print('If we didnt go into the first if clause, we shouldnt be here either. Check!')
                print(new_order)
            filtered_chans = filtered_chans[new_order]
            new_data = new_data[new_order]
        # save data to file
        feat_link = '/home/emil/EmoCog/data/temporary/' + str(row['Patient']) + str(row['Day']) + 'feat_filtered.pkl'
        with open(feat_link, "wb") as f:
            print('pickling filtered feat data...')
            pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        new_df_col.append(feat_link)
        new_chan_col.append(filtered_chans)
    common_df = common_df.assign(BinnedData=new_df_col, GoodChans=new_chan_col)
    return common_df
