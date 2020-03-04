#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Load in resampled, processed data and compute power spectra for each day
#(use 4 min bins so have 30fps and each half second is 1 hour of the day)
import numpy as np
import pandas as pd
from scipy import stats
from scipy import signal
#import mne
import h5py
import os
import matplotlib
matplotlib.use('Agg') #turn off displaying figures
import matplotlib.pyplot as plt
import argparse
import gc
import subprocess
import pdb

for patient_code in ['cb46fd46']:
    for day in [4]:
        ##Load the data
        # patient_code='a0f66459'
        # day=3
        data_dir='/data2/users/stepeter/Preprocessing/' #load in filtered data
        ecog_fn = data_dir+'processed_' + patient_code + '_' + str(day) + '.h5'
        fin = h5py.File(ecog_fn,"r")

        #Make save directory if it does not exist
        saveDir='/home/emil/powSpecTest/'+patient_code+'_'+str(day)+'/'
        if not os.path.exists(saveDir):
            print('hello')
            os.makedirs(saveDir)

        ###me
        start = 29413*500
        ###
        Fs=fin['f_sample'][()]
        tstep=Fs*100 #30 minute bins
        Nbins=16*36-6 #enough bins for whole day, starting at 8AM
        #Nbins = 20
        binRemainder=0 #no remainder
        goodChanInds=fin['goodChanInds'][()]

        #Compute power spectra for each time bin
        tmpDat=fin['dataset'][0,:]
        datLen=len(tmpDat)
        good_data = fin['dataset'][goodChanInds,start:]
        print(good_data.shape)
        really_good = np.ones(good_data.shape[0]).astype('bool')
        really_good[30]=False
        really_good[-1]=False
        good_data =good_data[really_good,:]
        #n_fft = 2048
        #win = signal.windows.hann(n_fft, True)
        for i in range(Nbins):
            fig, ax1 = plt.subplots(1, 1, figsize=(10,6))
            print('Bin #'+str(i+1)+' of '+str(Nbins))
            if datLen<(int((i+1)*tstep-1)):
                print('skipping...')
            else:
                data=[]
            #     if i==(Nbins-1):
            #         data = fin['dataset'][goodChanInds,int(i*tstep):-1] #go to the end of the data
            #     else:
                data = good_data[:,int(i*tstep):int((i+1)*tstep-1)]
                #freqT, Pxx = signal.welch(data, fs=Fs, nfft=n_fft, window=win) #nperseg=4096)
                freqT, Pxx = signal.welch(data, fs=Fs,window='hanning') #nperseg=4096)
                freq=np.zeros([len(freqT),1])
                freq[:,0]=freqT
            #     pdb.set_trace()
                ax1.semilogy(freq*np.ones([1,Pxx.shape[0]]), np.transpose(Pxx)) # scale to dB
                del freq
                del Pxx
                gc.collect()
                ax1.set_ylim((1e-3,1e4))
                ax1.set_title('Power spectra for window {} till {}'.format(i*tstep,(i+1)*tstep-1))
            strFile = '/home/emil/powSpecTest/'+patient_code+'_'+str(day)+'/'+'testPowerSpec_'+patient_code+'_day'+str(day)+'_bin_'+str(i)+'.png'
            if os.path.isfile(strFile):
                os.remove(strFile)   # Opt.: os.system("rm "+strFile)
            plt.savefig(strFile,bbox_inches='tight')
            plt.close(fig)
            del fig
            del ax1
            gc.collect()
            #map function for parallel processing
        vidlink = saveDir+'test.mp4'
        if os.path.isfile(vidlink):
            os.remove(vidlink)   # Opt.: os.system("rm "+strFile)
        #Use Linux command line command of ffmpeg to create video
        #return_code = subprocess.call('ffmpeg -i '+saveDir+'PowerSpec_'+patient_code+'_day'+str(day)+'_bin_%d.png -vcodec mpeg4 '+saveDir+'specVid.mp4', shell=True) 
        return_code = subprocess.call('ffmpeg -i '+saveDir+'testPowerSpec_'+patient_code+'_day'+str(day)+'_bin_%d.png -vcodec mpeg4 '+vidlink, shell=True) 
        # !ffmpeg -i /data2/users/stepeter/Preprocessing/powSpecTest/PowerSpec_a0f66459_day3_bin_%d.png -vcodec mpeg4 /data2/users/stepeter/Preprocessing/powSpecTest/test.mp4


# In[16]:


np.random.seed(5)
lel = np.random.normal(1,4,(3,2))


# In[17]:


mel = np.random.normal(1,4,(3,2))


# In[19]:


lel


# In[20]:


mel

