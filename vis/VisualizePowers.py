

#Load in resampled, processed data and compute power spectra for each day
#(use 4 min bins so have 30fps and each half second is 1 hour of the day)
import numpy as np
from scipy import signal
import pandas as pd
import h5py
import os
import matplotlib
matplotlib.use('Agg') #turn off displaying figures
import matplotlib.pyplot as plt
import gc
import subprocess



if __name__ == '__main__':
    for patient_code in ['d6532718']:
        for day in [5]:#,6,7,8]:
            ##Load the data
            data_dir=os.path.join('/nas/ecog_project/derived/processed_ecog/', patient_code, 'full_day_ecog')
            ecog_fn = os.path.join(data_dir, patient_code+ '_fullday_' + str(day) + '.h5')
            fin = h5py.File(ecog_fn,"r")
            ## Load good chans
            chan_inf = pd.read_hdf(ecog_fn, key='chan_info')
            try:
                chan_inf['LTP1']  # does it exist?
                chan_inf.loc['goodChanInds', 'LTP1'] = 0  # set to bad
                print('WARNING. SET ELECTRODDE LTP1 MANUALLY TO BAD CHANNEL. DO YOU WANT THIS?')
            except KeyError:
                None


            #Make save directory if it does not exist
            saveDir='/home/emil/powSpecTest/'+patient_code+'_'+str(day)+'/'
            if not os.path.exists(saveDir):
                print('hello')
                os.makedirs(saveDir)

            ###me
            start = 7*3600*500
            ###
            Fs=fin['f_sample'][()]
            tstep=Fs*100 #30 minute bins
            Nbins=16*36-6 #enough bins for whole day, starting at 8AM
            #Nbins = 20
            binRemainder=0 #no remainder

            goodChanInds = chan_inf.loc['goodChanInds'].values.astype(bool)

            #Compute power spectra for each time bin
            tmpDat=fin['dataset'][0,:]
            datLen=len(tmpDat)
            good_data = fin['dataset'][goodChanInds,start:]
            print(good_data.shape)
            really_good = np.ones(good_data.shape[0]).astype('bool')
            # really_good[62]=False
            # really_good[-1]=False
            good_data =good_data[really_good, :]
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
                    if i+1 == 1:
                        # what has the highest power at around 80Hz?
                        print('lalelele')
                    if len(freqT.shape) == 2:
                        freqT = freqT.flatten()
                        print('ayoouuus')
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
