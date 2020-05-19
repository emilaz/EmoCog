import util.sync_utils as sutil
from feature_related.feature_data_holder import FeatDataHolder
from label_related.label_data_holder import LabelDataHolder
import pickle
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool


def _generate_raws_single_day(patient, day, overwrite=False):
    if not overwrite:
        try:
            ret = _load_raws_single_day(patient, day)
            print('Loaded data for patient {}, day {}'.format(patient, day))
        except FileNotFoundError:
            print('Data not on File. Generating now...')
            overwrite = True
    if overwrite:
        path_ecog, path_vid = sutil.find_paths(patient, day)
        realtime_start, realtime_end = sutil.find_start_and_end_time(path_vid)  # output in secs from midnight
        if realtime_start < 7 * 3600:  # if it's before 7AM, reset it to 7 AM
            realtime_start = 7 * 3600
        if realtime_end > 23 * 3600:
            realtime_end = 23 * 3600  # if it's after 23PM, reset to 23PM
        if realtime_start >= realtime_end:
            print('For patient {}, day {}, starting time was after end time. Returning None'.format(patient, day))
            return None
        print('Day {}, start time is {} , end time is {}'.format(day, realtime_start, realtime_end))
        feat_data = FeatDataHolder(path_ecog, realtime_start, realtime_end)
        label_data = LabelDataHolder(path_vid, realtime_start, realtime_end, col='Happy')
        # save feat data
        feat_link, label_link = save_raws_single_day(patient, day,
                                                     feat_data.get_bin_data(),
                                                     label_data.get_pred_bin(),
                                                     overwrite=True)
        ret = [patient, day, realtime_start, realtime_end, feat_link,
               label_link, np.array([f.upper() for f in feat_data.chan_labels])]

        # ret = [patient, day, realtime_start, realtime_end, feat_data.get_bin_data(),
        #        label_data.get_pred_bin(), feat_data.chan_labels]
        data_link = '/home/emil/EmoCog/data/raws/' + str(patient) + str(day) + '.pkl'
        with open(data_link, 'wb') as f:
            pickle.dump(ret, f, protocol=pickle.HIGHEST_PROTOCOL)
        del feat_data
        del label_data
        print('Results done. day {} patient {}'.format(day, patient))
    return ret


def generate_raws(patients, days):
    pat_day_df = pd.DataFrame(columns=['Patient', 'Day'], data=zip(patients, days)).explode('Day')
    p = Pool(8)
    res = p.starmap(_generate_raws_single_day, pat_day_df.values)
    return res


def _load_raws_single_day(patient, day):
    """
    Function for unpickling raw data info from file
    Input: Patient, day
    Output: Array consisting of patient, day, start and end time,
            link to feature data, link to label data, good channels]
    """
    data_link = '/home/emil/EmoCog/data/raws/' + str(patient) + str(day) + '.pkl'
    with open(data_link, 'rb') as f:
        ret = pickle.load(f)
    return ret


def save_raws_single_day(patient, day, feat_data, label_data, overwrite = False):
    feat_link = '/home/emil/EmoCog/data/raws/' + str(patient) + str(day) + 'feat.pkl'
    label_link = '/home/emil/EmoCog/data/raws/' + str(patient) + str(day) + 'label.pkl'
    if not overwrite:  # in this case, just check if the files already exist
        exist_feat = os.path.exists(feat_link)
        exist_label = os.path.exists(label_link)
        if not (exist_feat and exist_label):
            overwrite = True
            print('Either raw feats or labels were not found. Overwriting although you told me not to.')
    if overwrite:
        with open(feat_link, "wb") as f:
            print('pickling feat data...')
            pickle.dump(feat_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(label_link, "wb") as f:
            print('pickling label data...')
            pickle.dump(label_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return feat_link, label_link


if __name__ == '__main__':
    # feed in list of patients and days we want processed
    patients = ['cb46fd46','af859cc5']
    days = [[3,4,5,6,7],[3,4,5]]
    generate_raws(patients,days)