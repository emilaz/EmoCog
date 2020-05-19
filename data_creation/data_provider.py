from feature_related.feature_generator import FeatureGenerator
from label_related.label_generator import LabelGenerator
import pandas as pd
import util.data_utils as dutil
import util.label_utils as lutil
import sys
import itertools
from data_creation.create_raws import generate_raws
from data_creation.data_processing import process, process_test


"""
Class that brings together feature and label side to provide all data needed for classification.
Synchronizes the data.
Data should be generated and analysed here, then saved to file using the DataUtil class.
"""


class DataProvider:
    """
    Init function. Start and end time are set for a time period I checked manually to be more or less okay.
    Creates classes to hold label and feature data in memory and classes to create feats and labels.
    Input: draw bool, if we want to visualize the happy/non-happy ratio et al.
    """
    def __init__(self, draw=False):
        self.is_loaded = False  # bool to check whether the raw data has already been loaded into memory
        self.draw = draw
        self.all_days_df = None
        self.featuregen = None
        self.lablegen = None

    def _load_raws(self, patients, days, filter_additional=None):
        # new: create df out of patient/days for using dask then
        res = generate_raws(patients, days)
        print('Got single day results, putting it into one DataFrame.')
        columns = ['Patient', 'Day', 'Start', 'End', 'BinnedData', 'BinnedLabels', 'GoodChans']
        all_days_df = pd.DataFrame(res, columns=columns)
        # due to size limitations, the workers only return paths to data on disk. replace with actual data here
        all_days_df = (all_days_df.sort_values(['Patient', 'Day'])).reset_index(drop=True)
        self.all_days_df = all_days_df
        print('Creating the generators..')
        self.featuregen = FeatureGenerator(all_days_df, filter_additional=filter_additional)
        self.lablegen = LabelGenerator(all_days_df)
        print('done.')
        # das hier erstmal nicht. spaeter zur analyse vielleicht wieder
        # self.annotsgen = LabelGenerator(all_days_df,mask=self.featuregen.bad_indices['NaNs'])
        self.is_loaded = True

    def reload_generators(self):
        self.featuregen = FeatureGenerator(self.all_days_df)
        self.lablegen = LabelGenerator(self.all_days_df)

    def create_unprocessed_data(self, configs):
        """ Function to generate the feats and labels, given the input hyperparas
        Input:  Configs, i.e. Windowsize, sliding window, start and end (in s), train bool, variance to be explained,
                cutoff if classification.
        Output: Features, Labels
        """
        # train data
        if 'expvar' not in configs.keys():
            configs['expvar'] = 95
        # check whether train or test data, set start and end sample accordingly
        x_df = self.featuregen.generate_features(wsize=configs['wsize'], sliding_window=configs['sliding'])
        y_df= self.lablegen.generate_labels(wsize=configs['wsize'], sliding_window=configs['sliding'])
        joined_df = x_df.merge(y_df, on=['Patient','Day'])
        new_col_order = ['Patient','Day','X', 'BadIndices', 'Y', 'Ratio']
        joined_df = joined_df.reindex(columns=new_col_order)
        # joined_df.to_hdf('/home/emil/data/check_me_out.hdf', key='df')
        # print('saved')

        # annots, _ = self.annotsgen.generate_labels(configs['wsize'], start=start,end=end, sliding_window=configs['sliding'])
#         if self.draw:
#             LabelVis.plot_happy_ratio(y,rat)
#             LabelVis.plot_happy_ratio(annots,_)
#             preds = y[~np.isnan(y)] ### this is for the confusion matrix between human annotations and openface labels
#             annots = annots[~np.isnan(y)]
#             preds = preds[~np.isnan(annots)]
#             annots = annots[~np.isnan(annots)] ###
#             ClassificationVis.conf_mat(preds, annots)

        return joined_df

    def get_data(self, configs, reload=False):
        # if data already exists, simply reload
        if reload:
            try:
                x_tr, y_tr, x_ev, y_ev = dutil.load_data_from_file(configs)
                print('Loading Data from File..done')
            except FileNotFoundError:  # file doesn't exist
                print('Data not on File.')
                return
        else:
            print(' Loading raw data into memory...')
            if not self.is_loaded:
                self._load_raws(configs['patient'], configs['days'])
            print('Now generating the actual data..')
            joined_df = self.create_unprocessed_data(configs)
            x_tr, y_tr, x_ev, y_ev = process(joined_df,
                                             self.featuregen.good_channels, configs)
            print('Done. Saving to file for later use.')
            dutil.save_data_to_file(x_tr, y_tr, x_ev, y_ev, configs)
        # now do the cutoff
        if 'cutoff' in configs.keys(): 
            cutoff = configs['cutoff']
            print('Doing cutoff')
            y_tr = lutil.do_cutoff(y_tr, cutoff)
            y_ev = lutil.do_cutoff(y_ev, cutoff)
        return x_tr, y_tr, x_ev, y_ev


    def get_test_data(self, configs, path_to_procesing):
        # First, load processing tools (standardization, common channels, PCA model etc.)
        tools = dutil.load_processing_tools_from_path(path_to_procesing)
        print(' Loading raw data into memory...')
        self._load_raws(configs['patient'], configs['days'], filter_additional=tools['GoodChans'])
        print('Now generating the actual data..')
        joined_df = self.create_unprocessed_data(configs)
        x, y = process_test(joined_df, configs, tools)
        # print('Done. Saving to file for later use.')
        # dutil.save_data_to_file(x_tr, y_tr, x_ev, y_ev, configs)
        # now do the cutoff
        if 'cutoff' in configs.keys():
            cutoff = configs['cutoff']
            print('Doing cutoff')
            y = lutil.do_cutoff(y, cutoff)
        return x, y


if __name__ == '__main__':

    provider = DataProvider()

    patient = ['af859cc5']
    days = [[2,3,4,5]]
    # patient = ['cb46fd46', 'af859cc5']
    # days = [[3, 4, 5, 6, 7], [3, 4, 5]]
    wsize = 100
    sliding = 25
    shuffle = False
    expvar = 90
    ratio = .8
    configs = dict()
    configs['patient'] = patient
    configs['days'] = days
    configs['wsize'] = wsize
    configs['sliding'] = sliding
    configs['expvar'] = expvar
    configs['ratio'] = ratio
    configs['shuffle'] = shuffle
    print('los', configs)
    #provider.reload_generators()
    provider.get_data(configs)
    # muell = provider.get_test_data(configs,
    #                                "/home/emil/EmoCog/data/new_labels/pca_models/"
    #                                "patient_['cb46fd46', 'af859cc5']_days_[[3, 4, 5, 6, 7],"
    #                                " [3, 4, 5]]_wsize_100_sliding_25_expvar_90_ratio_0.8_shuffle_False.pkl")

    #
    wsizes = [100, 50, 30, 5]
    shuffle = [True, False]
    combos = itertools.product(wsizes, shuffle)
    configs['sliding'] = False
    for c in combos:
        configs['wsize'] = c[0]
        configs['shuffle'] = c[1]
        provider.get_data(configs)

    dutil.remove_temporary_data()
    sys.exit(0)