from FeatureRelated.feature_generator import Feature_generator
from LabelRelated.label_generator import Label_generator 
from FeatureRelated.feature_data_holder import FeatDataHolder
from LabelRelated.label_data_holder import LabelDataHolder
import pandas as pd
import util.data_utils as dutil
import util.label_utils as lutil
import util.sync_utils as sutil
from data_processing import process

import numpy


"""
Class that brings together feature and label side to provide all data needed for classification.
Synchronizes the data.
This class is not designed to be used elsewhere. Data should be generated and analysed here, then saved to file using the DataUtil class.
"""


class DataProvider:
    """
    Init function. Start and end time are set for a time period I checked manually to be more or less okay.
    Creates classes to hold label and feature data in memory and classes to create feats and labels.
    Input: draw bool, if we want to visualize the happy/non-happy ratio et al.
    """
    def __init__(self, draw = False):
        self.is_loaded = False #bool to check whether the raw data has already been loaded into memory
        self.draw = draw
        self.all_days_df = None
        self.featuregen = None
        self.lablegen = None
        
    def _load_raws(self, patient, days):
        if type(days) is int:
            days = [days]
        #this dataframe saves pat,day,st,end,the raw, non-standardized, non-PCA features and corresponding labels
        all_days_df = pd.DataFrame(columns = ['Patient','Day','Start','End','BinnedData','BinnedLabels', 'GoodChans'], index=range(len(days)))
        for enum,day in enumerate(days):
            print(day,'this day')
            curr_ret = self.load_raws_single_day(patient,day)    
            all_days_df.loc[enum] = curr_ret
        all_days_df = (all_days_df.sort_values(['Day'])).reset_index(drop=True)
        self.all_days_df = all_days_df
        self.featuregen = Feature_generator(all_days_df)
        self.lablegen = Label_generator(all_days_df)
        #das hier erstmal nicht. spaeter zur analyse vielleicht wieder
        #self.annotsgen = Label_generator(all_days_df,mask=self.featuregen.bad_indices['NaNs']) #this is for conf mat later
        self.is_loaded = True
        
    
    def load_raws_single_day(self, patient,day):
        path_ecog, path_vid = sutil.find_paths(patient,day)
        realtime_start, realtime_end = sutil.find_start_and_end_time(path_vid) #output in secs from midnight
        if realtime_start<7*3600: #if it's before 7AM, reset it to 7 AM
            realtime_start=7*3600
        if realtime_end>23*3600:
            realtime_end= 23*3600
        print('Day {}, start time is {} , end time is {}'.format(day,realtime_start,realtime_end))
        feat_data = FeatDataHolder(path_ecog,realtime_start, realtime_end)
        label_data = LabelDataHolder(path_vid, realtime_start, realtime_end, col = 'Happy')
        ret = [patient, day,realtime_start,realtime_end,feat_data.get_bin_data(),label_data.get_pred_bin(), feat_data.chan_labels]
        del(feat_data)
        del(label_data)
        return ret


    def reload_generators(self):
        self.featuregen = Feature_generator(self.all_days_df)
        self.lablegen = Label_generator(self.all_days_df)
        


    """
    Function to generate the feats and labels, given the input hyperparas
    Input: Configs, i.e. Windowsize, sliding window, start and end (in s), train bool, variance to be explained, cutoff if classification.
    Output: Features, Labels
    """
    def generate_data(self, configs):
        #train data
        if 'expvar' not in configs.keys():
            configs['expvar']=95
        #check whether train or test data, set start and end sample accordingly
        x = self.featuregen.generate_features(wsize = configs['wsize'], sliding_window=configs['sliding'])
        y,rat = self.lablegen.generate_labels(wsize = configs['wsize'], sliding_window=configs['sliding'])
        #annots, _ = self.annotsgen.generate_labels(wsize=configs['wsize'], start=start,end=end, sliding_window=configs['sliding'])
#         if self.draw:
#             LabelVis.plot_happy_ratio(y,rat)
#             LabelVis.plot_happy_ratio(annots,_)
#             preds = y[~np.isnan(y)] ### this is for the confusion matrix between human annotations and openface labels
#             annots = annots[~np.isnan(y)]
#             preds = preds[~np.isnan(annots)]
#             annots = annots[~np.isnan(annots)] ###
#             ClassificationVis.conf_mat(preds, annots)
        x_tr, y_tr, x_ev, y_ev = process(x, y, self.featuregen.get_bad_indices(), self.featuregen.df['GoodChans'].loc[0], configs)
        return x_tr, y_tr, x_ev, y_ev
    
    
    
    def get_data(self, configs, shuffle_data = False, reload = False):
        #if data already exists, simply reload
        if reload:
            try:
                x_tr,y_tr,x_ev, y_ev = dutil.load_data_from_file(configs)
                print('Loading Data from File..done')
            except FileNotFoundError: #file doesn't exist
                print('Data not on File.')
                return
        else:
            print(' Loading raw data into memory...')
            if not self.is_loaded:
                self._load_raws(configs['patient'],configs['days'])
            print('And creating the data..')
            x_tr, y_tr, x_ev, y_ev = self.generate_data(configs)
            print('Done. Saving to file for later use.')
            dutil.save_data_to_file(x_tr,y_tr,x_ev, y_ev, configs)
        #now do the cutoff
        if 'cutoff' in configs.keys(): 
            cutoff = configs['cutoff']
            print('Doing cutoff')
            y_tr = lutil.do_cutoff(y_tr, cutoff)
            y_ev = lutil.do_cutoff(y_ev, cutoff)
        return x_tr, y_tr, x_ev, y_ev


#TODO das hier muss wahrsch. modifiziert werden jetzt wo es als python file rennen soll
if __name__ == '__main__':

    provider = DataProvider()

    patient = 'cb46fd46'
    days = [3,4,5,6,7]
    wsize = 100
    sliding = 25
    expvar = 90
    ratio = .8
    shuffle = False
    configs =dict()
    configs['patient']=patient
    configs['days']=days
    configs['wsize']=wsize
    configs['sliding']=sliding
    configs['expvar'] = expvar
    configs['ratio'] = ratio
    configs['shuffle']= shuffle
    print('los')
    #provider.reload_generators()
    muell = provider.get_data(configs)


    # wsize = 50
    # sliding = False
    #
    # configs['wsize']=wsize
    # configs['sliding']=sliding
    #
    # print('los')
    # muell = provider.get_data(configs)
    #
    #
    # shuffle = True
    #
    # configs['shuffle']=shuffle
    #
    # print('los')
    # muell = provider.get_data(configs)
    #
    #
    # wsize = 100
    # shuffle = False
    #
    #
    # configs['wsize']=wsize
    # configs['shuffle']=shuffle
    #
    # print('los')
    # muell = provider.get_data(configs)
    #
    #
    # shuffle = True
    # configs['shuffle']=shuffle
    #
    # print('los')
    # muell = provider.get_data(configs)
    #
    #
    # wsize = 5
    # shuffle = False
    #
    # configs['wsize']=wsize
    # configs['shuffle']=shuffle
    #
    # print('los')
    # muell = provider.get_data(configs)
    #
    # shuffle = True
    # configs['shuffle']=shuffle
    #
    # print('los')
    # muell = provider.get_data(configs)
    # del(provider)
    # del(muell)
    #
