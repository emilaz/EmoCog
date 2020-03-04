#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#%matplotlib inline
import sys
sys.path.append('..')
import numpy as np
from scipy import signal
import warnings
import util.feature_utils as util


# In[ ]:


"""
Class to generate features. Uses preprocessed data held in memory by FeatDataHolder class.
"""
class Feature_generator:
    
    """
    Init function.
    """
    def __init__(self,df):
        #sampling frequency and last sample taken
        self.sfreq = 500 #will always be, hopefully lol
        #TODO CHange this back to use the one from featureutils!!!!
        self.df = util.filter_common_channels(df)
        self.bad_indices = None #this needs to be passed on to the label side. Will include bad indices found during calculation

        
    """
    Function needed for calculating the features. Central piece on the feature side. Works as follows:
    columns: channels, for each channel the 150 frequencies (0-150Hz) (hece freq*cha length), binned logarithmically
    Rows: Time steps, defined by sliding window+window size
    resulting matrix is 2D, Time Stepsx(Freq*Channels)
    In case of generating train data, this function also saves mean and stddev for standardization purpose.
    Input: Start and end time (in secs), bool for whether train data or not (for PCA), window size and sliding window in sec
    Output: Standardized, binned data.
    """
    def _generate_features_single_day(self,data, wsize = 100, sliding_window=False):
        bads = []
        time_it = 0
        mat = None
        idx = 0
        while True:
            stop = time_it + wsize
            if stop > data.shape[1]:
                print('This day gave us {} seconds worth of data'.format(data.shape[1]-1))
                break
            #Note that each column is exactly one second.
            #get data in range of ALL channels
            curr_data = data[:,time_it:stop,:].reshape(data.shape[0],-1)
            #welch method 
            fr,psd = signal.welch(curr_data,self.sfreq,nperseg=250)
            
            #if there are nans in the psd, something's off. throw away, save index, continue
            if np.isnan(psd).any():
                bads +=[idx] #current index baad
                if (sliding_window):
                    time_it += sliding_window
                else:
                    time_it += wsize
                idx+=1
                continue
            
            fr_bin,psd_bin = util.bin_psd(fr,psd)
            idx+=1
            if mat is None:
                #first time. create first column, flatten w/o argument is row major 
                mat = psd_bin.flatten()
            else:
                #after, add column for each time step
                mat = np.column_stack((mat,psd_bin.flatten()))
            #sliding window?
            if (sliding_window):
                time_it += sliding_window
            else:
                time_it += wsize
        return mat, bads #we do the standardization after the filtering
    
    
    def generate_features(self, wsize = 100, sliding_window=False):
        #here, check how many days we need for the requested datasize
        time_passed = 0
        curr_data = None
        for day in self.df['Day']:
            print('Day', day)
            data = self.df[self.df['Day']==day].BinnedData.values[0]
            mat, bad = self._generate_features_single_day(data, wsize, sliding_window)
            if curr_data is None:
                curr_data = mat
                self.bad_indices = np.array(bad)
            else:
                self.bad_indices = np.append(self.bad_indices,np.array(bad)+len(self.bad_indices)+curr_data.shape[1])
                if mat is not None:
                    curr_data = np.append(curr_data,mat,axis=1)
        return curr_data
            

    """
    Function to return the bad indices found by filtering. Important: First filter out the nan indices, then the artifacts!
    Order is important
    Output: Dictionary of bad data points.
    """
    
    def get_bad_indices(self):
        return self.bad_indices
        


# def filter_common_channels(common_df):
#     good_idx = util.find_common_channels(common_df[['Day','GoodChans']])
#     for idx,day in enumerate(good_idx['Day']):
#         good = good_idx.loc[good_idx['Day']==day,'CommonChans'][idx]
#         new_data = common_df[common_df['Day']==day]['BinnedData'][idx][good,:]
#         common_df.loc[common_df['Day']==day,'BinnedData'] = [new_data]
#         spraa = common_df.loc[common_df['Day']==day,'GoodChans'][idx][good]
#         common_df.loc[common_df['Day']==day,'GoodChans'] = [spraa]
#     return common_df


# In[ ]:


# days = [3,4]
# all_days_df = pd.DataFrame(columns = ['Patient','Day','BinnedData','BinnedLabels', 'GoodChans'], index=range(len(days)))
# for enum,day in enumerate(days):
#     print(day,'this day')
#     ####for testing
#     labels = np.arange(day,10)[:,None]
#     features = np.tile(np.arange(day,10)[None,:],(3,1)).astype('float')
#     features[:,2]=np.nan
#     print(features)
#     good_chans = np.array(['yes'+str(day),'mes','tes'])
#     curr_ret = ['test', day, features,labels,good_chans]      
#     all_days_df.loc[enum] = curr_ret


# In[ ]:


# gen = Feature_generator(all_days_df)
# gen.generate_features(wsize =1)


# In[ ]:


##### A WHOLE LOT OF PLOTTING FUNCTIONS


# wut=pecog.generate_features(0,12500,expl_variance=90)

# # print(wut.shape)


# pecog.vis_pc()

# pecog.curr_data.shape


# lel=pecog.curr_data.T
# med=np.median(lel.reshape(-1,8,lel.shape[1]),axis=0)
# men=np.mean(lel.reshape(-1,8,lel.shape[1]),axis=0)
# print(lel.shape)
# for i in range(8):
#     plt.plot(lel[8*8+i,:], label='Bin %d' %i)
# plt.legend()
# plt.xlabel('Time Window')
# plt.ylabel('PSD')
# plt.title('Welch Transformation results')
# plt.show()
# for i in range(8):
#     plt.plot(men[i,:], label='Bin %d' %i)
# plt.legend()
# plt.xlabel('Time Window')
# #plt.yscale('log')
# plt.ylabel('PSD')
# plt.title('Welch Transformation, Mean over Channels - Standardized ')
# plt.show()

# for i in range(8):
#     plt.plot(med[i,:], label='Bin %d' %i)
# plt.legend()
# plt.xlabel('Time Window')
# #plt.yscale('log')
# plt.ylabel('PSD')
# plt.title('Welch Transformation, Median over Channels - Standardized')
# plt.show()



# lel=pecog.temp_mat.T
# med=np.median(lel.reshape(-1,8,lel.shape[1]),axis=0)
# men=np.mean(lel.reshape(-1,8,lel.shape[1]),axis=0)
# print(lel.shape)
# for i in range(8):
#     plt.plot(lel[8*8+i,:], label='Bin %d' %i)
# plt.legend()
# plt.xlabel('Time Window')
# plt.ylabel('PSD')
# plt.title('Welch Transformation results')
# plt.show()
# for i in range(8):
#     plt.plot(men[i,:], label='Bin %d' %i)
# plt.legend()
# plt.xlabel('Time Window')
# #plt.yscale('log')
# plt.ylabel('PSD')
# plt.title('Welch Transformation, Mean over Channels ')
# plt.show()

# for i in range(8):
#     plt.plot(med[i,:], label='Bin %d' %i)
# plt.legend()
# plt.xlabel('Time Window')
# #plt.yscale('log')
# plt.ylabel('PSD')
# plt.title('Welch Transformation, Median over Channels')
# plt.show()

# f=h5py.File('/data2/users/stepeter/Preprocessing/processed_cb46fd46_4.h5')

# sprr=f['dataset'][()]

# for i in range(8,14):
#     plt.plot(sprr[i,125*100*500:127*100*500])
# plt.xlabel('t')
# plt.ylabel('uV')
# plt.show()
# for i in range(10,12):
#     plt.plot(pecog.data[i,125*100*500:127*100*500])
# plt.xlabel('t')
# plt.ylabel('uV')
# plt.show()

# for i in range(8,14):
#     plt.plot(sprr[i,10*100*500:12*100*500])
# plt.xlabel('t')
# plt.ylabel('uV')
# plt.title('One Patient, Sample Channels')

# for i in range(25,28):
#     plt.plot(pecog.data[i,35*100*500:40*100*500])
# plt.xlabel('t')
# plt.ylabel('uV')
# plt.show()

# for i in range(25,28):
#     plt.plot(pecog.data[i,38*100*500:40*100*500])
# plt.xlabel('t')
# plt.ylabel('uV')
# plt.show()

# for i in range(25,28):
#     plt.plot(pecog.data[i,38*100*500:50*100*500])
# plt.xlabel('t')
# plt.ylabel('uV')
# plt.show()

# np.argmax(lel.reshape(-1,8,lel.shape[1])[:,1,34])

# pecog.generate_features()

# ttest=np.abs(pecog.pca.transform(np.eye(pecog.curr_data.shape[1])))

# ttest_sum[:16]

# ttest.shape

# #how much did each bin contribute?
# ttest_sum=ttest.sum(axis=1)
# ttest_sum=ttest[:,0]
# ttest_shaped=ttest_sum.reshape(pecog.data.shape[0],-1)
# cont_bins=ttest_shaped.sum(axis=0)
# cont_elecs=ttest_shaped.sum(axis=1)
# plt.figure(figsize=(7,3))
# plt.bar(np.arange(len(cont_bins)),cont_bins)
# ticks=['[0,1]','(1,2]','(2,4]','(4,8]','(8,16]','(16,32]','(32,64]','(64,150]']
# plt.xticks(np.arange(len(cont_bins)),ticks)
# plt.title('Contributions of Bins to PD 0')
# plt.xlabel('Bins')
# plt.ylabel('Absolute Contribution')
# plt.show()

# plt.figure(figsize=(3,15))
# plt.barh(np.arange(len(cont_elecs)),cont_elecs)
# plt.yticks(np.arange(len(cont_elecs)),list(pecog.chan_labels))
# plt.ylabel('Region')
# plt.xlabel('Absolute Contribution')
# plt.title('Contributions of Chans to PD')

# mni_file=pd.read_excel('/data2/users/stepeter/mni_coords/cb46fd46/cb46fd46_MNI_atlasRegions.xlsx')
# for en,i in enumerate(pecog.chan_labels):
#     print(en)
#     if( sum(mni_file['Electrode'].isin([i]))==0):
#         print(i)
#         print(en)
        

# print(mni_file['Electrode'].loc[7])

# pecog.chan_labels.shape

# pc1=np.sum(np.abs(ttest[:,1].reshape(-1,8)),axis=1)
# pc2=np.sum(np.abs(ttest[:,2].reshape(-1,8)),axis=1)
# pc3=np.sum(np.abs(ttest[:,3].reshape(-1,8)),axis=1)
# pc4=np.sum(np.abs(ttest[:,4].reshape(-1,8)),axis=1)

# mni_coords_fullfile='/data2/users/stepeter/mni_coords/cb46fd46/cb46fd46_MNI_atlasRegions.xlsx'
# plot_ecog_electrodes_mni_from_file_and_labels(mni_coords_fullfile,pecog.chan_labels,num_grid_chans=64, colors=cont_elecs[:-1])

# plot_ecog_electrodes_mni_from_file_and_labels(mni_coords_fullfile,pecog.chan_labels,num_grid_chans=64, colors=pc1[:-1])
# plot_ecog_electrodes_mni_from_file_and_labels(mni_coords_fullfile,pecog.chan_labels,num_grid_chans=64, colors=pc2[:-1])
# plot_ecog_electrodes_mni_from_file_and_labels(mni_coords_fullfile,pecog.chan_labels,num_grid_chans=64, colors=pc3[:-1])
# plot_ecog_electrodes_mni_from_file_and_labels(mni_coords_fullfile,pecog.chan_labels,num_grid_chans=64, colors=pc4[:-1])

