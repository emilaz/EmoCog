
# coding: utf-8

# In[17]:


from FeatureRelated.feature_generator import Feature_generator
from LabelRelated.label_generator import Label_generator 
from FeatureRelated.feature_data_holder import FeatDataHolder
from LabelRelated.label_data_holder import LabelDataHolder
import numpy as np
from Vis import LabelVis, ClassificationVis


# In[18]:


class DataProvider:

    def __init__(self, col = 'Happy_predicted', draw = False):
        realtime_start = 29413 #this is 8h, 10min and 13s into day 4
        video_start = realtime_start - 29344 # 29344 is the beginning of recordings of video data (see /home/emil/data/sync_data)
        self.duration = 37820+4500 #in seconds, of course
        self.feat_data = FeatDataHolder('/data2/users/stepeter/Preprocessing/processed_cb46fd46_4.h5',start=realtime_start, duration=self.duration)
        self.label_data = LabelDataHolder('/home/emil/data/hdf_data/cb46fd46_7_imp_columns.hdf',video_start,video_start+self.duration, col = col )
#        self.label_data = LabelDataHolder('/home/emil/data/hdf_data/cb46fd46_7_imp_columns.hdf',video_start, col= col)
        self.featuregen = Feature_generator(self.feat_data)
        self.lablegen = Label_generator(self.label_data,mask=self.featuregen.mask_bin)
        self.draw = draw

    #get datasets
    def get_data(self,wsize=100,sliding=10,s_sample=0,e_sample=None, train=True, expl_var=95, cutoff = None):
        if e_sample == None:
            e_sample = self.duration-1
        #train data
        x = self.featuregen.generate_features(start=s_sample,end=e_sample,expl_variance=expl_var,train=train,sliding_window=sliding)
        y,rat = self.lablegen.generate_labels(start=s_sample,end=e_sample, sliding_window=sliding, cutoff=cutoff)
        if self.draw:
            LabelVis.plot_happy_ratio(y,rat)
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        return x,y


# In[15]:


# #this is to find cutoff

# bla = DataProvider(draw=True)
# ma = DataProvider(draw=True, col ='annotated')

# y = bla.get_data(sliding=10)
# my = ma.get_data(sliding=10)

# thresh = .3
# for thresh in [.2]:
#     y_class = y.copy()
#     my_class = my.copy()
#     y_class_nans = np.isnan(y_class)
#     my_class_nans = np.isnan(my_class)
#     print(y_class_nans)
#     y_class[y_class<thresh]=0
#     y_class[y_class>0]=1
#     y_class[y_class_nans]=np.nan
#     my_class[my_class<thresh]=0
#     my_class[my_class>0]=1
#     my_class[my_class_nans]=np.nan

#     LabelVis.plot_nan_ratio(my_class)
#     LabelVis.plot_nan_ratio(y_class)

#     ynonan =y_class[~my_class_nans]
#     mynonan = my_class[~my_class_nans]
#     mynonan = mynonan[~np.isnan(ynonan)]
#     ynonan = ynonan[~np.isnan(ynonan)]

#     ClassificationVis.conf_mat(mynonan,ynonan)

