#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


"""
This class is used to generate usable labels from the data preprocessed by the LabelDataHolder class. 
Class was initially created to avoid reloading data into memory every time functions were altered/debugged.
"""

class Label_generator:
    
    """
    Init function.
    Input: DF with label data of (possibly) several days
    """
    def __init__(self,df):
        self.fps=30
        self.df=df



    """
    Generates labels. Use sliding window if features are also generated with sliding window. 
    Note that we specified a time frame in the LabelDataHolder class. Within that frame, we can specify start and end point for the labels generated.
    Useful for splitting into train/test data.
    Input: Start point, End Point (in seconds), Windowsize for subsuming, method for subsuming labels in given window (ratio or median), Cutoff if we need boolean labels).
    Output: Usable labels for regression/classification task. If classification is used, also returns the percentage of non-nan labels per label generated.
    """
    def _generate_labels_single_day(self,data, start=0, end=None, wsize = 100, sliding_window = 0,method='ratio'):
        if method != 'ratio' and method != 'median':
            raise NameError('The given method does not exist. Try one of the following: ratio,median.')
        if method is 'median':
            print('Note: The median method is currently a 75 percentile.')
        if end is None or end >= data.shape[0]:
            end = data.shape[0]-1
        #average "happiness" per second
        happy_portion = np.nanmean(np.array(data, dtype = 'float'),axis = 1)
        #check nans along the 31FPS
        non_nans_per_s = np.count_nonzero(~np.isnan(np.array(data, dtype='float')),axis = 1)
        #if(sliding_window):
        self.labels = []
        good_ratio = []
        time_it = start
        while True:
            stop = time_it+wsize
            curr_data = happy_portion[time_it:stop]           
            curr_non_nans = np.sum(non_nans_per_s[time_it:stop])
            if not curr_data.size:
                print('Whole chunk of NaNs. Check this again.')
                if sliding_window:
                    time_it += sliding_window
                else:
                    time_it += wsize
                if time_it + wsize > end:
                    break
                continue
            #here, we divide by len(curr_data), because we don't want the influence of nans that were thrown away due to bad feature points.
            good_ratio += [float(curr_non_nans)/(self.fps*len(curr_data))]
            if method =='ratio':
                self.labels += [np.nanmean(curr_data)]
            elif method == 'median':
                self.labels += [np.nanpercentile(curr_data,q=75)]
            if sliding_window:
                time_it += sliding_window
            else:
                time_it += wsize
            if time_it + wsize > end:
                break
        self.labels = np.array(self.labels)
        return self.labels, good_ratio #THIS IS NOT USED SO FAR, BUT SHOULD BE.
    
    
    def generate_labels(self, start, end, wsize = 100, sliding_window =0, method='ratio'):
        dur = end - start
        time_passed = 0
        curr_data = None
        ratio = None
        idx = 0
        while dur>time_passed+wsize:
            try:
                day = self.df['Day'].loc[idx]
            except KeyError:
                print ("Not enough data loaded into memory for this request.")
                return data
            curr_dur = self.df['End'].loc[idx]-self.df['Start'].loc[idx]
            if start + wsize>= curr_dur: #if startsample is after duration of data of first day, go to next day, change stuff
                end = end-curr_dur+min(0,start-curr_dur) 
                start = max(start-curr_dur,0) #sometimes, start<curr_dur, aber kein ganzes window passt mehr rein.
                print('jo soviel vergangen{}, so sind die nun {},{}'.format(passed_not_used,start,end))
                continue
            data = self.df['BinnedLabels'].loc[idx]
            mat, rat = self._generate_labels_single_day(data,start,start+dur-time_passed, wsize, sliding_window)
            if idx == 0:
                curr_data = mat
                ratio = rat
            else:
                curr_data = np.append(curr_data,mat,axis = None)
                ratio = np.append(ratio,rat,axis=None)
            idx +=1
            if sliding_window:
                time_passed = wsize+sliding_window*(curr_data.shape[0]-1)
            else:
                time_passed = wsize*(curr_data.shape[0])
            start = 0 #for the next day, in case the initial starting time wasn't zero
        return curr_data, ratio


# In[ ]:





# In[ ]:


# indices=np.array([  161,   162,   163,   173,   174,   175,   176,   177,   178,179,   180,   181,   182,   228,   229,   230,   231,   232,          233,   234,   235,   236,   259,   260,   261,   262,   263,          264,   265,   266,   267,   268,   269,   270,   271,   272,          273,   274,   275,   276,   277,   278,   279,   280,   281,          282,   283,  1284,  1285,  1286,  1287,  1288,  1289,  1290,         1291,  1472,  1473,  1474,  1475,  1907,  1908,  1909,  1910,         2150,  2151,  2152,  2153,  2154,  2409,  2410,  2411,  2412,
#          2413,  2414,  2415,  2416,  2417,  2418,  2419,  2420,  2780,         2781,  2782,  2997,  2998,  2999,  3000,  3001,  3002,  3003,         3004,  3005,  3029,  3030,  3031,  3032,  3033,  3034,  3035,         3036,  3037,  3443,  3444,  3445,  3451,  3452,  3453,  3486,         3487,  3488,  3489,  3490,  3491,  3492,  3493,  3494,  3651,         3652,  3653,  3654,  3655,  3656,  3657,  3658,  3659,  3688,         3689,  3690,  3691,  3692,  3693,  3694,  3710,  3711,  3712,         3713,  3714,  3715,  3716,  3717,  3718,  3719,  3720,  3729,
#          3730,  3731,  3732,  3733,  3734,  3735,  3736,  3737,  3738,         3739,  3740,  3741,  3742,  3743,  3744,  3745,  3746,  3747,         3748,  3749,  3751,  3752,  3753,  3754,  4280,  4281,  4282,         4283,  4284,  4285,  4286,  4287,  8187,  8188,  8189,  8190,         8191,  8192,  8193,  8194, 12643, 12644, 12645, 12646, 12647,        12661, 12662, 12663, 12664, 22924, 22925, 22926, 22956, 22957,        22958, 22969, 22970, 22971, 22999, 23000, 23001, 23003, 23004,        23005, 27302, 27303, 27304, 28218, 28219, 28220, 29225, 29226,
#         29227, 29228, 29229, 29230, 29231, 29232, 29233, 29234, 29235,        29236, 29237, 29238, 29239, 29240, 29241, 29242, 29243, 29244,        29245, 29246, 29247, 29248, 29271, 29272, 29273, 29274, 29275,        29276, 29277, 29278, 29279, 29280, 29281, 29282, 29283, 29284,        29285, 29286, 29287, 29288, 29289, 29290, 29291, 29292, 29293,        29294, 29295, 29296, 29297, 29298, 29299, 29300, 29301, 29302,        29303, 29304, 29305, 29306, 29307, 29308, 29309, 29310, 29311,        29312, 29313, 29314, 29315, 29316, 29317, 29318, 29319, 29320,
#         29321, 29322, 29323, 29324, 29325, 29326, 29327, 29328, 29329,        29330, 29331, 29332, 29333, 29334, 29335, 29336, 29337, 29338,        29339, 29340, 29341, 29342, 29343, 29344, 29345, 29346, 29347,        29348, 29349, 29350, 29351, 29352, 29382, 29383, 29384, 29385,        29386, 29387, 29388, 29389, 29390, 29391, 29392, 29393, 29394,        29395, 29396, 29397, 29398, 29399, 29400, 29401, 29402, 29403,        29404, 29405, 29406, 29407, 29408, 29409, 29410, 29411, 29494,        29495, 29496, 29502, 29503, 29504, 29505, 29506, 29507, 29508,
#         29509, 29510, 29511, 29512, 29513, 29514, 35518, 35519, 35520,        35696, 35697, 35698, 35861, 35862, 35863, 36151, 36152, 36153,        36154, 36155, 36156, 36157, 36158, 36159, 36160, 36161, 36162,        36163, 36164, 36165, 36166, 36297, 36298, 36299, 37687, 37688, 37689])
# mask=np.zeros(45000)
# mask[indices]=1


# In[ ]:


# stop=12500
# start=11
# test=Label_generator('/home/emil/data/hdf_data/cb46fd46_8_imp_columns.hdf', start=start,stop=stop, wsize=100)
# meds,meds_rat=test.generate_labels(method='median',mask=mask)
# mea,mea_rat=test.generate_labels(method='ratio',mask=mask)
# meds_sl, meds_sl_rat=test.generate_labels(method='median', sliding_window=True,mask=mask)
# mea_sl,mea_sl_rat=test.generate_labels(method='ratio',sliding_window=True,mask=mask)

# meds_cl, meds_cl_rat=test.generate_labels(method='median', classification=True, cutoff=.1,mask=mask)
# mea_cl, mea_cl_rat=test.generate_labels(method='ratio',classification=True,cutoff=.1,mask=mask)


# In[ ]:


#import matplotlib.pyplot as plt
# #plot the nan ratio
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


# plt.scatter(range(1400),test.df['Happy_predicted'].values[:1400], s=2)
# plt.title('Raw HappyFace Predictions')
# plt.xlabel('t')
# plt.ylabel('Happy (binary)')


# In[ ]:


#test=Label_generator('/home/emil/data/hdf_data/cb46fd46_8_imp_columns.hdf',start=11,stop=43205)

# mas=test.generate_labels(start=0, end=30000,method='ratio',mask=None)

# # plt.plot(np.mean(test.pred_bin,axis=0))
# # plt.xlabel('sec')
# # plt.ylabel('Happy prediction')


# # plt.plot(test.labels)
# # plt.xlabel('window')
# plt.ylabel('Happy prediction')

