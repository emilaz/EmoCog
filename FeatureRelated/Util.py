
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.decomposition import PCA


# In[ ]:


def standardize(data,ax=0):
    data_mean = np.mean(data,axis=ax)
    data_dem = data-data_mean
    std = np.std(data,axis=ax)
    data_stand = data_dem/std
    return data_stand

#this function caps at 150Hz, then bins the data in a logarithmic fashion to account for smaller psd values in higher freqs
def bin_psd(fr,psd):
    fr_trun=fr[fr<=150]
    fr_total=len(fr_trun)
    fr_bins=np.arange(int(np.log2(max(fr_trun))+1))
    #truncate everythin above 150Hz
    psd=psd[:,fr<=150]
    psd_bins=np.zeros((psd.shape[0],len(fr_bins)))
    prev=0
    #these are the general upper limits. they don't give info where in fr/psd these frequencies acutally are!
    max_psd_per_bin=np.exp2(fr_bins).astype('int')
    #hence we need this method:
    prev=0
    limits=np.zeros((max_psd_per_bin.shape[0],2),dtype='int')
    for en,b in enumerate(max_psd_per_bin):
        if en==0:
            arr=np.where((fr_trun >=prev)&(fr_trun<=b))[0]
        else:
            arr=np.where((fr_trun >prev)&(fr_trun<=b))[0]
        check=np.array([min(arr),max(arr)])
        limits[np.log2(b).astype('int')]=check
        prev=b
    prev=0
    for b in fr_bins:
        if (b==fr_bins[-1] or limits[b][1]>=fr_total):
            psd_bins[:,b]+=np.sum(psd[:,limits[b,0]:],axis=1)
        else:
            psd_bins[:,b]=np.sum(psd[:,limits[b,0]:limits[b,1]+1],axis=1)
    return fr_bins, psd_bins
    
        
def get_no_comps(data,expl_var_lim):
    comps=min(100,min(data.shape))
    pca=PCA(n_components=comps)
    pca.fit(data)
    tot=0
    for idx,c in enumerate(pca.explained_variance_ratio_):
        tot+=c
        if c>expl_var_lim:
            return idx+1
    return pca.n_components_
        


# In[ ]:


na=PCA(n_components=20)

