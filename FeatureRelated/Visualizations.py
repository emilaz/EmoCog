
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt


# In[4]:


def vis_raw_data(data, start_sec, stop_sec, freq, chans=None):
    if chans is None:
        chans=range(data.shape[0])
    st=int(start_sec*freq)
    stp=int(stop_sec*freq)
    data=data[chans,st:stp]
    for p in range(0,len(chans)-1):
        plt.plot(data[p])
    plt.show()

#currently not usable (deprecated)
# def vis_welch_data(start,stop,chans=None):
#     #account for wsize
#     start=int(start/self.wsize)
#     stop=int(stop/self.wsize)
#     rem=self.curr_data[:,start:stop]
#     plt.imshow(rem,cmap='viridis',aspect='auto')


def vis_pc(pca,data):
    for p in range(pca.n_components):
        plt.plot(pca.transform(data)[:,p])
    plt.xlabel('Time (in w_size)')
    plt.ylabel('PC Value')
    plt.title('First %d principal components' % pca.n_components)
    plt.show()
    
#get elbow curve. This also outputs the optimal n_components for the given desired explained variancce.
def __elbow_curve(datapart,expl_var_lim):
    components = range(1, datapart.shape[1] + 1)
    explained_variance = []
    #till where?
    lim=min(100, datapart.shape[1])
    count=0
    for component in tqdm(components[:lim]):
        pca = PCA(n_components=component)
        pca.fit(datapart)
        expl_var=sum(pca.explained_variance_ratio_)
        explained_variance.append(expl_var)
        count+=1
        if(expl_var>(expl_var_lim/100.)):
            optimal_no_comps=count
            break
    if(explained_variance[-1:][0]<(expl_var_lim/100.)):
        print('Could not explain more than %d %% of the variance. n_comps is set to match this. Consider increasing data range or lowering demanded explained variance' % expl_var*100)
        optimal_no_comps=components[-1:]
    sns_plot = sns.regplot(
        x=np.array(components[:count]), y=explained_variance,
        fit_reg=False).get_figure()
    return optimal_no_comps

