from FeatureRelated.feature_generator import FeatureGenerator
from LabelRelated.label_generator import LabelGenerator
from FeatureRelated.feature_data_holder import FeatDataHolder
from LabelRelated.label_data_holder import LabelDataHolder
import pandas as pd
import util.data_utils as dutil
import util.label_utils as lutil
import util.sync_utils as sutil
from multiprocessing import Pool
from data_processing import process


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
        
    def _load_raws(self, patient, days):
        # new: create df out of patient/days for using dask then
        pat_day_df = pd.DataFrame(columns=['Patient','Day'], data = zip(patient,days)).explode('Day')
        # this dataframe saves pat,day,st,end,the raw, non-standardized, non-PCA features and corresponding labels
        columns = ['Patient', 'Day', 'Start', 'End', 'BinnedData', 'BinnedLabels', 'GoodChans']
        #####
        # all_days_df = pd.DataFrame(columns=columns, index=range(len(days)))
        # for enum, day in enumerate(days):
        #     print(day, 'this day')
        #     curr_ret = self.load_raws_single_day(patient,day)
        #     all_days_df.loc[enum] = curr_ret
        #####
        # new: do dask stuff
        # results = []
        # for paras in pat_day_df.values:
        #     res = dask.delayed(self.load_raws_single_day)(*paras)
        #     results.append(res)
        # res = dask.compute(*results)
        ###
        # even newer: do this with multiprocessing. too many core dumps happening with dask
        p = Pool(8)
        res = p.starmap(self.load_raws_single_day, pat_day_df.values)
        all_days_df = pd.DataFrame(res,columns=columns)
        all_days_df = (all_days_df.sort_values(['Patient', 'Day'])).reset_index(drop=True)
        #for now, save this to file
        self.all_days_df = all_days_df
        self.featuregen = FeatureGenerator(all_days_df)
        self.lablegen = LabelGenerator(all_days_df)
        # das hier erstmal nicht. spaeter zur analyse vielleicht wieder
        # self.annotsgen = LabelGenerator(all_days_df,mask=self.featuregen.bad_indices['NaNs'])
        self.is_loaded = True
        
    def load_raws_single_day(self, patient, day):
        path_ecog, path_vid = sutil.find_paths(patient, day)
        realtime_start, realtime_end = sutil.find_start_and_end_time(path_vid)  # output in secs from midnight
        if realtime_start < 7*3600:  # if it's before 7AM, reset it to 7 AM
            realtime_start = 7*3600
        if realtime_end > 23*3600:
            realtime_end = 23*3600 #if it's after 23PM, reset to 23PM
        print('Day {}, start time is {} , end time is {}'.format(day, realtime_start, realtime_end))
        feat_data = FeatDataHolder(path_ecog, realtime_start, realtime_end)
        label_data = LabelDataHolder(path_vid, realtime_start, realtime_end, col='Happy')
        ret = [patient, day, realtime_start, realtime_end, feat_data.get_bin_data(),
               label_data.get_pred_bin(), feat_data.chan_labels]
               #  'TESTTEST', feat_data.chan_labels]
        del feat_data
        del label_data
        return ret

    def reload_generators(self):
        self.featuregen = FeatureGenerator(self.all_days_df)
        self.lablegen = LabelGenerator(self.all_days_df)
        
    """
    Function to generate the feats and labels, given the input hyperparas
    Input:  Configs, i.e. Windowsize, 
            sliding window, 
            start and end (in s), 
            train bool, 
            variance to be explained, 
            cutoff if classification.
    Output: Features, Labels
    """
    def generate_data(self, configs):
        # train data
        if 'expvar' not in configs.keys():
            configs['expvar'] = 95
        # check whether train or test data, set start and end sample accordingly
        x_df = self.featuregen.generate_features(wsize=configs['wsize'], sliding_window=configs['sliding'])
        y_df= self.lablegen.generate_labels(wsize=configs['wsize'], sliding_window=configs['sliding'])
        joined_df = x_df.merge(y_df, on = ['Patient','Day'])

        # annots, _ = self.annotsgen.generate_labels(configs['wsize'], start=start,end=end, sliding_window=configs['sliding'])
#         if self.draw:
#             LabelVis.plot_happy_ratio(y,rat)
#             LabelVis.plot_happy_ratio(annots,_)
#             preds = y[~np.isnan(y)] ### this is for the confusion matrix between human annotations and openface labels
#             annots = annots[~np.isnan(y)]
#             preds = preds[~np.isnan(annots)]
#             annots = annots[~np.isnan(annots)] ###
#             ClassificationVis.conf_mat(preds, annots)
        x_tr, y_tr, x_ev, y_ev = process(joined_df,
                                         self.featuregen.good_channels, configs)
        return x_tr, y_tr, x_ev, y_ev
    
    
    def get_data(self, configs, shuffle_data=False, reload=False):
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
            print('And creating the data..')
            x_tr, y_tr, x_ev, y_ev = self.generate_data(configs)
            print('Done. Saving to file for later use.')
            dutil.save_data_to_file(x_tr, y_tr, x_ev, y_ev, configs)
        # now do the cutoff
        if 'cutoff' in configs.keys(): 
            cutoff = configs['cutoff']
            print('Doing cutoff')
            y_tr = lutil.do_cutoff(y_tr, cutoff)
            y_ev = lutil.do_cutoff(y_ev, cutoff)
        return x_tr, y_tr, x_ev, y_ev


#TODO das hier muss wahrsch. modifiziert werden jetzt wo es als python file rennen soll
if __name__ == '__main__':

    provider = DataProvider()

    patient = ['cb46fd46','af859cc5']
    days = [[3],[4]]
    # patient = ['cb46fd46']
    # days = [[3, 4]]
    wsize = 100
    sliding = False
    expvar = 90
    ratio = .8
    shuffle = False
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
    muell = provider.get_data(configs)

    # provider = DataProvider()
    # patient = ['cb46fd46']
    # days = [[3,4]]
    # configs['patient'] = patient
    # configs['days'] = days
    # muell = provider.get_data(configs)



    # wsize = 50
    # sliding = False
    #
    # configs['wsize'] = wsize
    # configs['sliding'] = sliding
    #
    # print('los', configs)
    # muell = provider.get_data(configs)
    # #
    # #
    # # shuffle = True
    # #
    # # configs['shuffle']=shuffle
    # #
    # # print('los', configs)
    # # muell = provider.get_data(configs)
    # #
    # #
    # wsize = 100
    # # shuffle = False
    # #
    # #
    # configs['wsize'] = wsize
    # # configs['shuffle']=shuffle
    # #
    # print('los', configs)
    # muell = provider.get_data(configs)
    # #
    # #
    # # shuffle = True
    # # configs['shuffle']=shuffle
    # #
    # # print('los', configs)
    # # muell = provider.get_data(configs)
    # #
    # #
    # wsize = 5
    # # shuffle = False
    # #
    # configs['wsize'] = wsize
    # # configs['shuffle']=shuffle
    # #
    # print('los', configs)
    # muell = provider.get_data(configs)
    # #
    # # shuffle = True
    # # configs['shuffle']=shuffle
    # #
    # # print('los', configs)
    # # muell = provider.get_data(configs)
    # # del(provider)
    # # del(muell)
    # #
    #
    # wsize = 30
    # configs['wsize'] = wsize
    # print('los', configs)
    # muell = provider.get_data(configs)