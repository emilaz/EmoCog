
# coding: utf-8

# In[ ]:


import pandas as pd
from FeatureRelated.feature_generator import Feature_generator
from LabelRelated.label_generator import Label_generator 
from FeatureRelated.feature_data_holder import FeatDataHolder
from LabelRelated.label_data_holder import LabelDataHolder
import numpy as np


# In[2]:


#get datasets
def get_data(wsize=100,sliding=10,s_sample=0,e_sample=12500, s_sample_ev=9000, e_sample_ev=12000):
    
    feat_data = FeatDataHolder('/data2/users/stepeter/Preprocessing/processed_cb46fd46_4.h5')
    label_data = LabelDataHolder('/home/emil/data/hdf_data/cb46fd46_8_imp_columns.hdf',feat_data.start,feat_data.end)


    #edf_path, emo_path=get_paths(pat_name,sess)
    featuregen = Feature_generator(feat_data,wsize=wsize)
    lablegen = Label_generator(label_data,mask=featuregen.mask_bin,wsize=wsize)

    x = featuregen.generate_features(start=s_sample,end=e_sample,expl_variance=95,train=True,sliding_window=sliding)

    y,rat = lablegen.generate_labels(start=s_sample,end=e_sample, sliding_window=sliding, classification=True)

    x_ev = featuregen.generate_features(start = s_sample_ev,end = e_sample_ev,train = False,sliding_window = sliding)
    y_ev, y_ev_rat = lablegen.generate_labels(start = s_sample_ev,end = e_sample_ev,sliding_window = sliding,classification=True)
    
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    x_ev = x_ev[~np.isnan(y_ev)]
    y_ev = y_ev[~np.isnan(y_ev)]

    return x,y,x_ev,y_ev

def get_data_from_file(wsize=100,sliding=10,s_sample=0,e_sample=12500, s_sample_ev=9000, e_sample_ev=12000):
    link='./data/ws_'+str(wsize)+'_str_'+str(sliding)+'_tr'+'_s_'+str(s_sample)+'_e_'+str(e_sample)+'_ev_'+'s_'+str(s_sample_ev)+'_e_'+str(e_sample_ev)+'.hdf'
    df = pd.read_hdf(link)
    x = df['x'][0]
    y = df['y'][0]
    x_ev = df['x_ev'][0]
    y_ev = df['y_ev'][0]
    
    return x,y,x_ev,y_ev

def save_data_to_file(x,y,x_ev,y_ev,wsize=100,sliding=10,s_sample=0,e_sample=12500, s_sample_ev=9000, e_sample_ev=12000):
    link='./data/ws_'+str(wsize)+'_str_'+str(sliding)+'_tr'+'_s_'+str(s_sample)+'_e_'+str(e_sample)+'_ev_'+'s_'+str(s_sample_ev)+'_e_'+str(e_sample_ev)+'.hdf'
#     link_ev='./data/ws'+str(wsize)+'sl'+str(sliding)+'s'+str(s_sample_ev)+'e'+str(e_sample_ev)+'_ev.hdf'
    #save stuff to file:
    df = pd.DataFrame(data=[[x,y,x_ev,y_ev]],columns=['x','y','x_ev','y_ev'])
#     df_tr.to_hdf(link_tr,key='df')
#     df_ev = pd.DataFrame(data=[[x_ev,y_ev]],columns=['x_ev','y_ev'])
#     df_ev.to_hdf(link_ev,key='df')
#     df_ev = pd.DataFrame(data=[[x_ev,y_ev]],columns=['x_ev','y_ev'])
    df.to_hdf(link,key='df')

