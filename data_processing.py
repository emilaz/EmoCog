import numpy as np
import pandas as pd
import util.sync_utils as util
import util.feature_utils as futil
import util.data_utils as dutil
from sklearn.decomposition import PCA



"""
Sets up PCA parameters in case of training, otherwise just transforms
Input: Data, train bool, amount of variance one wants to be explained (automatically calculates no. of PD needed)
Output: Data in PC space.
"""
def setup_PCA(curr_data,expl_variance, pca = None):
    if pca is None:
        print('Setting up PCA on current data range...')
        no_comps=futil.get_no_comps(curr_data,expl_variance)
        pca=PCA(n_components=no_comps)
        pca.fit(curr_data)
    return pca, pca.transform(curr_data)


def shuffle_data(x,y, ratio):
    print('shuffling the data')
    train_idx = np.random.choice(len(y),int(ratio*len(y)),replace=False)
    train = np.zeros(len(y),dtype='bool')
    train[train_idx] = 1
    print(x.shape,y.shape, 'shapes bby')
    y_tr = y[train]
    x_tr = x[:,train]
    x_ev = x[:,~train]
    y_ev = y[~train]
    return x_tr,y_tr,x_ev,y_ev


def apply_processing(x,y, good_chans, other_configs):
    tools = dutil.load_processing_tools(other_configs)
    pca = tools['Model']
    std_lim, std_mean, std, mean = filter_channels_for_standardizing(tools, good_chans)
    print(std_lim.shape, std_mean.shape)
    print(std.shape,mean.shape)
    print(x.shape)
    artifacts,  _, _ = futil.detect_artifacts(x, std_lim, std_mean)
    x, y = util.filter_artifacts(x, y, artifacts)
    print('After artfilter', x.shape)
    x = futil.standardize(x,std,mean)
    x = pca.transform(x.T)
    return x, y
    

def filter_channels_for_standardizing(tools, other_good_chans):
    artifact_paras = tools['Artifact Parameter']
    standar_paras = tools['Standardization Parameter']
    orig_good_chans = tools['GoodChans']
    filter_art = pd.DataFrame(columns=['Day','GoodChans','BinnedData'])
    filter_art.loc[0] = [0,orig_good_chans,artifact_paras[0][:,None,None]]
    filter_art.loc[1] = [1,orig_good_chans,artifact_paras[1][:,None,None]]
    ret_art = futil.filter_common_channels(filter_art, other_good_chans)
    std_lim = ret_art.loc[0]['BinnedData'].squeeze()
    std_mean = ret_art.loc[1]['BinnedData'].squeeze()
    
    filter_stan = pd.DataFrame(columns=['Day','GoodChans'])
    filter_stan.loc[0] = [0,orig_good_chans]
    filter_stan.loc[1] = [1,orig_good_chans]
    common = futil.find_common_channels(filter_stan['GoodChans'], other_good_chans)
    good_idx = futil.find_common_channel_indices(filter_stan, common)
    std = standar_paras[0][np.repeat(good_idx['CommonChans'].loc[0],8)]
    mean = standar_paras[1][np.repeat(good_idx['CommonChans'].loc[0],8)]
    return std_lim, std_mean, std, mean


def process(x,y, bad_indices, good_chans, configs):
    #first, synchronize this shit
    x_clean, y_clean = util.filter_nans(x,y,bad_indices)

    #next, separate into train and test
    #in theory, though very unlikely, due to the configs['sliding'] window,
    #there could still be train samples in the eval set as well.
    #filter those potential ones out
    if configs['sliding']:
        overlap = configs['wsize'] // configs['sliding'] -1 + bool(configs['wsize'] % configs['sliding'])
    else:
        overlap = 0

#     this shuffles things into train and test set
    if configs['shuffle']:
        x_tr,y_tr,x_ev,y_ev = shuffle_data(x_clean, y_clean, configs['ratio'])
    else:
        x_tr = x_clean[:,:int(configs['ratio'] * y_clean.shape[0])]
        y_tr = y_clean[:int(configs['ratio'] * y_clean.shape[0])]
        x_ev = x_clean[:,int(configs['ratio'] * y_clean.shape[0]) + overlap:]
        y_ev = y_clean[int(configs['ratio'] * y_clean.shape[0]) + overlap:]

    #next, filter out the artifacts
    artifacts_tr, std_lim, std_med = futil.detect_artifacts(x_tr) 
    artifacts_ev, _,_ = futil.detect_artifacts(x_ev, std_lim,std_med)
    x_tr, y_tr = util.filter_artifacts(x_tr,y_tr,artifacts_tr)
    x_ev, y_ev = util.filter_artifacts(x_ev,y_ev,artifacts_ev)
    #then, do standardizing 
    std = np.std(x_tr,axis=1)
    mean = np.mean(x_tr,axis=1)
    x_tr = futil.standardize(x_tr, std, mean)
    x_ev =futil.standardize(x_ev, std, mean)
    #and PCA on feature sets
    pca, x_tr = setup_PCA(x_tr.T,configs['expvar'])
    _, x_ev = setup_PCA(x_ev.T,configs['expvar'],pca)
    print('Saving PCA model and other processing info to file.')
    dutil.save_processing_tools(pca,[std_lim,std_med],[std,mean], good_chans, configs)
    #now ready to return/rumble
    return x_tr, y_tr, x_ev, y_ev
    

