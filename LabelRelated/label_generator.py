
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings


# In[ ]:


class Label_generator:
    def __init__(self,path,wsize=30,start=0,stop=None):
        if path.endswith('.csv'):
            self.df=pd.read_csv(path,error_bad_lines=False, low_memory=False)#,dtype={'realtime':'datetime64'})
        elif path.endswith('.hdf'):
            self.df=pd.read_hdf(path,error_bad_lines=False, low_memory=False)#,dtype={'realtime':'datetime64'})
        self.fps=31
        #account for 30FPS
        self.start=start*self.fps
        if stop is None:
            self.stop=self.df.shape[0]
        else:
            self.stop=stop*self.fps
        self.wsize=wsize
        self.df=self.df.iloc[self.start:self.stop]#,self.df.columns!= 'datetime']
        #self._convert_to_unix_time()
        self._bin_preds()
    
    #not needed atm    
#     def _convert_to_unix_time(self):        
#         #unix time in miliseconds
#         newcol = (pd.DatetimeIndex(self.df['realtime'])).astype(np.int64)//10**(6)
#         self.df=self.df.assign(unix_time=newcol)
    
    def _bin_preds(self):
        annot=self.df['Happy_predicted'].values
        #bin s.t. each column is one sec.
        end=annot.shape[0]//self.fps
        self.pred_bin=annot[:self.fps*end].reshape(self.fps,-1)

    #generates labels. Use sliding window if features are also generated with sliding window
    #if a classification method is used, we need a cutoff somewhere :)
    def generate_labels(self,start=0, end=None, mask=None, sliding_window= False,method='ratio', classification=False, cutoff=.07):
        if method!='ratio' and method!='median':
            raise NameError('The given method does not exist. Try one of the following: ratio,classification.')
        if mask is None:
            print('Warning. No filtering mask for bad data points was given. Assuming perfectly clean dataset.')
            mask=np.zeros(self.pred_bin.shape[1],dtype='bool')
        if end is None:
            end=self.pred_bin.shape[1]-1
        if end >= self.pred_bin.shape[1]:
            end=self.pred_bin.shape[1]-1
            print('Desired window too long. Setting to %d'% end)
        #average "happiness" per second
        happy_portion=np.nanmean(np.array(self.pred_bin, dtype='float'),axis=0)
        #check nans along the 31FPS
        non_nans_per_s=np.count_nonzero(~np.isnan(np.array(self.pred_bin, dtype='float')),axis=0)
        #if(sliding_window):
        self.labels=[]
        good_ratio=[]
        time_it=start
        while True:
            stop=time_it+self.wsize
            curr_mask=np.ma.compressed(np.ma.masked_array(range(time_it,stop),mask=mask[time_it:stop]))
            curr_data=happy_portion[curr_mask]
            curr_nans=np.sum(non_nans_per_s[curr_mask])
            if not curr_data.size:
                if sliding_window:
                    time_it+=1
                else:
                    time_it+=self.wsize
                if time_it+self.wsize > end:
                    break
                continue
            good_ratio+=[float(curr_nans)/(self.fps*len(curr_mask))]
            if method =='ratio':
                self.labels+=[np.nanmean(curr_data)]
            elif method =='median':
                self.labels+=[np.nanmedian(curr_data)]
            if sliding_window:
                time_it+=1
            else:
                time_it+=self.wsize
            if time_it+self.wsize > end:
                break
        self.labels=np.array(self.labels)
#         else:
#             #take average over windows size
#             end=(end-start)//self.wsize
#             sprt=happy_portion[start:start+end*self.wsize].reshape(self.wsize,-1)
#             #before applying mean, take only values that we want in this. use mask for that
#             mask=mask[start:start+end*self.wsize].reshape(self.wsize,-1)
#             masked_windows=np.ma.array(sprt, mask=mask)
#             if (method=='ratio'):
#                 self.labels=np.ma.compressed(np.ma.mean(masked_windows,axis=0))
#             elif method =='median':
#                 self.labels=np.ma.compressed(np.ma.median(masked_windows,axis=0))
#                 for col in range(len(masked_windows[0])):
#                     #print(np.ma.compressed(masked_windows[:,col]))
#                     plt.hist(np.ma.compressed(masked_windows[:,col]), bins=10)
#                     print(np.ma.median((masked_windows[:,col])))
#                     #print(np.ma.compressed(masked_windows[:,col]))
#                     plt.show()

        if(classification):
            self.labels[self.labels>cutoff]=1
            self.labels[self.labels<1]=0
            
        return self.labels #, good_ratio THIS IS NOT USED SO FAR, BUT SHOULD BE.


# In[ ]:


# stop=13200
# start=0
# test=Label_generator('/home/emil/data/hdf_data/cb46fd46_8_imp_columns.hdf',start=start,stop=stop, wsize=1800)
# meds,meds_rat=test.generate_labels(method='median')
# mea,mea_rat=test.generate_labels(method='ratio')
# meds_sl, meds_sl_rat=test.generate_labels(method='median', sliding_window=True)
# mea_sl,mea_sl_rat=test.generate_labels(method='ratio',sliding_window=True)

# meds_cl, meds_cl_rat=test.generate_labels(method='median', classification=True, cutoff=.1)
# mea_cl, mea_cl_rat=test.generate_labels(method='ratio',classification=True,cutoff=.1)


# In[ ]:


#plot the nan ratio
# br=np.unique(np.array(test.pred_bin, dtype='float'), return_counts=True)
# sum_nans=np.sum(br[1][2:])
# #print(sum_nans)
# vals=([str(br[0][0]),str(br[0][1]),str(br[0][2])],[br[1][0],br[1][1],sum_nans])
# print(vals[0],vals[1])
# plt.bar(vals[0],vals[1])
# plt.title("Occurences of 'Happy'/'Not Happy'/'N/A' predictions in %ds of data" % (stop-start))
# plt.xlabel('Prediction')
# plt.ylabel('Occurences')


# In[ ]:


#plot stuff
# plt.scatter(range(len(mea)),mea, label='Mean', c=mea_rat)
# plt.plot(range(len(mea)),mea, 'y--')
# plt.title('Windows of size %ds' %test.wsize)
# plt.ylabel('Value')
# plt.xlabel('Data point no.')
# cbar=plt.colorbar()
# cbar.set_label('Ratio Pred:NaN')
# plt.legend()
# plt.show()

# plt.scatter(range(len(meds)),meds, label='Median', c=meds_rat)
# plt.plot(meds, 'y--')
# cbar=plt.colorbar()
# cbar.set_label('Ratio Pred:NaN')
# plt.title('Windows of size %ds' %test.wsize)
# plt.ylabel('Value')
# plt.xlabel('Data point no.')
# plt.legend()
# plt.show()

# plt.scatter(range(len(meds_sl)),meds_sl, label='Median', c=meds_sl_rat)
# #plt.plot(meds_sl, 'b--')
# plt.ylabel('Value')
# plt.xlabel('Data point no.')
# plt.title('Values using sliding window of %ds' %test.wsize)
# plt.legend()
# cbar=plt.colorbar()
# cbar.set_label('Ratio Pred:NaN')
# plt.show()

# plt.scatter(range(len(mea_sl)),mea_sl, label='Mean', c=mea_sl_rat)
# #plt.plot(mea_sl, label='Mean')
# plt.ylabel('Value')
# plt.xlabel('Data point no.')
# plt.title('Values using sliding window of %ds' %test.wsize)
# plt.legend()
# cbar=plt.colorbar()
# cbar.set_label('Ratio Pred:NaN')
# plt.show()

# plt.plot(mea_cl,'b.', label='Mean',)
# plt.ylabel('Value')
# plt.xlabel('Data point no.')
# plt.title('Values using median and classification method, windowsize of %ds, threshold of %2.2f' % (test.wsize,.16))
# plt.legend()
# plt.show()

# plt.plot(meds_cl,'b.', label='Median',)
# plt.ylabel('Value')
# plt.xlabel('Data point no.')
# plt.title('Values using median and classification method, windowsize of %ds, threshold of %2.2f' % (test.wsize,.16))
# plt.legend()
# plt.show()


# In[ ]:


#test=Label_generator('/home/emil/data/hdf_data/cb46fd46_8_imp_columns.hdf',start=11,stop=43205)

# mas=test.generate_labels(start=0, end=30000,method='ratio',mask=None)

# # plt.plot(np.mean(test.pred_bin,axis=0))
# # plt.xlabel('sec')
# # plt.ylabel('Happy prediction')


# # plt.plot(test.labels)
# # plt.xlabel('window')
# plt.ylabel('Happy prediction')

