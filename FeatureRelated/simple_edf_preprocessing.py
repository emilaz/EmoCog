
# coding: utf-8

# In[1]:


import numpy as np
#import mne
import matplotlib.pyplot as plt
import h5py


# In[ ]:


class Preprocessor:
    def __init__(self,data=None,start_sample=0,end_sample=None):
        self.data=data
        self.start=start_sample
        if end_sample is None:
            self.end=data['dataset'].shape[1]-1
        else:
            self.end=end_sample
        self.df=data['dataset'][:,start_sample:end_sample]
        self.bad_idx=np.zeros(data['dataset'].shape[1],dtype='bool')
        self.bad_chans=np.zeros(data['dataset'].shape[0],dtype='bool')
        
    #what do I want to filter??
    #first, detect channels that are just flat signal
    def __bad_chan(self,prefiltered_sd_kurt=True):
        unique_entr_per_chan=np.count_nonzero(np.diff(np.sort(self.df)), axis=1)+1
        #which channels have only few entries? Throw away
        self.bad_chans=unique_entr_per_chan<10
        if(prefiltered_sd_kurt):
            #convert the goodchan indices to a bad chan boolean mask
            temp=np.ones(self.df.shape[0],dtype='bool')
            temp[self.data['goodChanInds']]=False
            self.bad_chans=np.logical_or(self.bad_chans,temp)
        else:
            all_std=self.data['SD_channels'][()]
            std_std=np.std(all_std)
            std_mean=np.mean(all_std)
            too_high_std=all_std>std_mean+1.5*std_std
            self.bad_chans=np.logical_or(self.bad_chans,too_high_std[0,:])
           
            
    #second, get indices of intermediate flat signal
    def __flat_sig(self):
        #entries with high altitude? 
        self.bad_idx[np.array(self.data['allChanArtifactInds'][()][:],dtype=int)]=True
        self.bad_idx=self.bad_idx[self.start:self.end] # these indices correspond to times starting from the beginning of recording. They need to be shifted so they match the 'new' beginning (start_sample)
#this function 
    def preprocess(self,prefiltered_sd_kurt=True,by_artifact=True):
        self.__flat_sig()
        self.__bad_chan(prefiltered_sd_kurt=prefiltered_sd_kurt)
        return self.df, self.bad_chans,self.bad_idx
        
    
#     #third, vals with too high of a variance are to be excluded. Note that varience is calculated without bad points
#     #for excluding high vars, we calculate the variance of each channel, calc the std across these variances, exclude values 
#     #that are more than 2 std away
#     def high_var(self):
#         #next, let's look at variation changes per channel
#         #get rid of points that are way off the mean
#         vars_per_chan=[]
#         for i in range(self.data.shape[0]):
#             if self.bad_chans[i]!=True:
#                 mean=np.mean(self.data[i,self.bad_idx[i]!=True])
#                 std=np.std(self.data[i,self.bad_idx[i]!=True])
#                 self.bad_idx[i]=np.logical_or(self.bad_idx[i],self.bad_idx[i]>mean+2*std)
#                 vars_per_chan+=[std**2]
#             else:
#                 vars_per_chan+=[np.nan]
#         #calc variance across time points that are OK
#         var=np.array(vars_per_chan)
#         #what is standart deviation and mean among the variances OF GOOD CHANNELS?
#         varmean=np.nanmean(var[self.bad_chans!=True])
#         varstd=np.std(var[self.bad_chans!=True])
#         print(varmean,varstd)
#         #get rid of avg that shoot way above. THESE ARE CHANNELS
#         self.bad_var=var>varmean+1*varstd

        


# In[ ]:


# df=h5py.File('/data2/users/stepeter/Preprocessing/processed_a0f66459_4.h5')
# pr=Preprocessor(df,start_sample=11*500,end_sample=43205*500)


# In[ ]:


# dat,tt,tw=pr.preprocess()


# In[ ]:


# print(dat.shape)
# print(tt.shape)
# tw.shape


# In[ ]:


# print(np.unique(tw,return_counts=True))


# In[ ]:


# print(tt.shape)
# print(tw.shape)
# print(dat.shape)
# aight=dat[tt!=True]
# aight=aight[:,tw!=True]
# print(aight.shape)
# print(tw[-1:])


# In[ ]:


# prepr.bad_var.shape#print(np.unique(prepr.bad_idx,coun))

# print(prepr.data[0].shape)
# print(prepr.bad_idx.shape)
# for p in range(0,127):
#     if not prepr.bad_chans[p]: 
#         if not prepr.bad_var[p]:
#             plt.plot(prepr.data[p,prepr.bad_idx[p]!=True])
#     #mark bad points
#     #plt.plot(np.arange(len(prepr.bad_idx[p]))[prepr.bad_idx[p]],np.zeros(len(prepr.bad_idx[p]))[prepr.bad_idx[p]]-.02, "-", markersize=1,)
# plt.legend()

# for p in range(0,127):
#     if (bad_chans)
#     plt.plot(prepr.data[p])

# plt.plot(tt[0])
# plt.plot(tt[1])
# plt.plot(peaks, tt[1][peaks], "x")


# np.random.seed(33)
# test=np.random.uniform(0.9,1,(1,100))
# print(test.shape)
# test[0,10:20]=0
# #test[1,15:25]=0
# test[0,40:73]=0.95
# #test[1,15:25]=0.95
# #peaks=argrelextrema(test,np.less,order=25)
# #plt.vlines(peaks,ymin=0, ymax=1)
# #print(peaks)
# grads=(np.gradient(test,axis=1)==0)
# print(np.zeros(len(grads[p]))[grads[p]].shape)
# for p in range(test.shape[0]):
#     plt.plot(test[p])
#     ym1 = np.ma.masked_where(grads[p] ==False , grads[p])
#     plt.plot(np.arange(len(grads[p])),ym1-1.02, "-")

# #np.argwhere(grads[1])[0]
# #print(grads)
# np.apply_along_axis(np.argwhere,1,grads)
# np.argwhere(grads[2])
# np.apply_along_axis(np.unique,1,grads)


# ee=np.eye(3)
# mask=np.zeros(ee.shape[0],dtype='bool')
# mask[1]=True
# mask[0]=True
# #print(mask)

# test=np.zeros(5)
# np.logical_or(np.ones(5),test==1)


# A=np.arange(5)
# A[np.arange(2,3)]=False
# print(A)

