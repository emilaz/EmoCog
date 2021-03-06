import sys

sys.path.append('..')
import numpy as np
import pandas as pd
import util.label_utils as util

"""
This class loads label data from file into memory and preprocesses it to be easily usable (binning etc.) for generating labels
"""


class LabelDataHolder:
    """
    Init function. Loads data in given start-end window and bins the data.
    Input:  Path to label file
            start and end point (in seconds) of labels wanted,
            type of label wanted (predicted vs. annotated)
    """

    def __init__(self, path, start, end, col='Happy'):
        if path.endswith('.csv'):
            self.df = pd.read_csv(path, error_bad_lines=False, low_memory=False)
        elif path.endswith('.hdf'):
            self.df = pd.read_hdf(path, error_bad_lines=False, low_memory=False)
        self.fps = 30
        self.start = start * self.fps
        self.end = end * self.fps
        empty_labels = self.create_empty_array()
        filled_labels = self.fill_label_array(empty_labels, col)
        labels = filled_labels[self.start:self.end]
        del filled_labels  # yes, this doesn't affect labels
        del empty_labels
        self.pred_bin = self._bin_preds(labels)

    """
    Creates an array for 24 hours of data. 
    Plan is to then fill in the labels at the corresponding points in time
    Cut this off at corresponding times later    
    """

    def create_empty_array(self):
        empty = np.empty(24 * 3600 * self.fps)  # for a whole of 24 hours
        empty[:] = np.nan
        return empty

    """
    Given the dataframe and desired number of frames, return array with labels.
    Input: 
    Output: Big array
    """

    def fill_label_array(self, labels, col):
        sess_nos = sorted(self.df['session'].unique())
        for sess in sess_nos:
            print('next sess', sess)
            curr_sess = self.df[self.df['session'] == sess]
            vid_nos = sorted(curr_sess['video'].unique())
            last_vid = vid_nos[-1]
            for vid in vid_nos:
                # actual frames saved in hdf (as opposed to supposed no. frames
                actual_frames = curr_sess[curr_sess['video'] == vid][col].values.astype(float)
                # wherever success = 0, we want to put nans. we don't know anything there. Hence we need flaot type
                success = curr_sess[curr_sess['video'] == vid]['success'].values
                actual_frames[success == 0] = np.nan
                if col == 'annotated':  # the annotated labels are strings. Convert here.
                    actual_frames = util.convert_labels_readable(actual_frames)
                start, supposed_no_frames = util.find_number_frames(curr_sess, vid,
                                                                    last_vid)  # frames the vid actually has
                ret = util.fill_frames(actual_frames, supposed_no_frames)  # fill the frames
                if start + supposed_no_frames > len(labels):
                    ret = ret[:len(labels) - start]
                    print('Omitting part of vid {}/{} because it goes beyond end time'.format(vid, last_vid))
                    # continue
                labels[start:start + supposed_no_frames] = ret
        return labels

    """
    Function for binning labels. Also converts the char predictions ('Happy'/'Not Happy') into usable bools if needed.
    Input: Column wanted
    Output: Binnned labels (one row = one sec)
    """

    def _bin_preds(self, labels):
        # bin s.t. each column is one sec.
        end = labels.shape[0] // self.fps
        ret = labels[:self.fps * end].reshape(-1, self.fps)
        return ret

    def get_pred_bin(self):
        return self.pred_bin
