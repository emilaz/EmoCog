
# coding: utf-8

# In[1]:


#%matplotlib inline
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import signal
import warnings
from Util import FeatureUtils as util


# In[2]:


#create a ECOG PCA class for its PCA object, hyperparas and other stuff
class Feature_generator:

    def __init__(self,dataholder):
        #sampling frequency and last sample taken
        self.sfreq = dataholder.sfreq
        self.data_bin, self.mask_bin = dataholder.get_bin_data_and_mask()
        self.pca = None
        self.std = None #these parameters are used for standardization.
        self.mean = None # Use same parameter and apply to eval/test data.


    #create matrix as follows:
    #columns: channels, for each channel the 200 frequencies (0-200Hz) (hece freq*cha length) BUT BINNED logarithmically
    #rows: Time steps
    #resulting matrix is 2D, Time Stepsx(Freq*Channels)
    #note that this matrix is prone to constant change. Save the current data as member variable
    #NEW: Option to do this with a sliding window of length wsize
    def _calc_features(self,time_sta,time_stp, train, wsize = 100, sliding_window=False):
        time_it = time_sta
        mat = None
        while True:
            stop = time_it + wsize
            if stop >= self.data_bin.shape[1]-1:
                print('Not enough data for set end %d. Returning all data that is available in given range.'% time_stp)
                break
                
            #Note that each column is exactly one second.
            #get data in range of ALL channels, applying the following mask to filter out seconds with a bad index
            mask = np.ma.compressed(np.ma.masked_array(range(time_it,stop),mask=self.mask_bin[time_it:stop]))
            curr_data = self.data_bin[:,mask,:].reshape(self.data_bin.shape[0],-1)
            
            #is this thing empty? continue
            if not curr_data.size:
                print('Warning. A whole chunck of %d s of data was thrown away here, starting at s %d. Check if this is correct' %(wsize,time_it))
                if(sliding_window):
                    time_it += sliding_window
                else:
                    time_it += wsize
                if time_it + wsize >= time_stp:
                    break
                continue
                
            #welch method 
            fr,psd = signal.welch(curr_data,self.sfreq,nperseg=250)
            fr_bin,psd_bin = util.bin_psd(fr,psd)
            if mat is None:
                self.fr_bin = fr_bin
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
            if time_it + wsize >= time_stp+1:
                break
        
        if train: #if it's train data, then get its mean and std for standardization
            self.std = np.std(mat.T,axis=0)
            self.mean = np.mean(mat.T,axis=0)

        data_scal = util.standardize(mat.T,self.std,self.mean)    
        return data_scal
    
    
    def _setup_PCA(self,curr_data,train,expl_variance):
        if not train and self.pca is None:
            raise ValueError('Train set has to be generated first, otherwise no principal axis available for data trafo.')
        if train:
            print('Setting up PCA on current data range...')
            no_comps=util.get_no_comps(curr_data,expl_variance)
            self.pca=PCA(n_components=no_comps)
            self.pca.fit(curr_data)
        return self.pca.transform(curr_data)
    
    def generate_features(self,start=0,end=None, wsize=100, sliding_window=False, train=True,expl_variance=85):
        curr_data=self._calc_features(time_sta=start, wsize=wsize, time_stp=end, train=train, sliding_window=sliding_window)
        princ_components=self._setup_PCA(curr_data,train=train,expl_variance=expl_variance)
        return princ_components
        
        


# In[ ]:


# pecog=Feature_generator('/data2/users/stepeter/Preprocessing/processed_cb46fd46_4.h5',prefiltered=False,wsize=100)

# for p in range(pecog.pca.n_components):
#     plt.plot(wut[:,p])
# plt.xlabel('Time (in w_size)')
# plt.ylabel('PC Value')
# plt.title('First %d principal components' % pecog.pca.n_components)
# plt.show()

# pecog.vis_raw_data(0,30000,range(20))

# #pecog.vis_raw_data(idx[0]-5,idx[1]+5)
# #pecog.vbis_raw_data(0,idx[1]+400)
# pecog.vis_welch_data(0,30000)



# good_data=pecog.calc_data_mat(idx[1]+100,idx[1]+400)
# pecog.pca.fit(good_data)
# good_data_trafo=pecog.pca.transform(good_data)
# print(good_data_trafo.shape)
# print(good_data.shape)


# print(good_data.shape)
# print(good_data_trafo.shape)

# comps=pecog.pca.components_
# print(pecog.raw.info['ch_names'][28])
# print(comps.shape)
# comps=comps.reshape((127,-1,2))
# print(np.argmax(comps[:,:,1],axis=1))
# #plt.plot(comps[5:,5:,0].T)
# plt.plot(comps[:,:,1].T)
# plt.ylim(-0.005,0.005)

# print(len(pecog.raw.info['chs']))

# #print(data_trafo)
# plt.plot(good_data_trafo[:,0])
# plt.plot(good_data_trafo[:,1])
# #plt.xlim(-0.00001,0.00001)
# #plt.ylim(-0.00001,0.00001)

# #max(data_trafo[:,1])-min(data_trafo[:,1])


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

