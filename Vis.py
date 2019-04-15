
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np


# In[4]:


def bar_plot_svc(scores_tr,scores_ev,gammas,thresholds = None):
    bar_width = 0.35
    plt.xticks(np.arange(0,20)+bar_width/2,range(0,20))
    bar1 = plt.bar(gammas,scores_tr,bar_width,label='Train',alpha=.5,color='b')
    bar2 = plt.bar(gammas+bar_width,scores_ev,bar_width,label='Ev',alpha=.5,color='r')
    plt.hlines(.5,0,15,linestyle='dashed',alpha=.2)
    #plt.plot(np.arange(0,15),[.5]*15,'--',color='k')
    plt.xlabel('Dim')
    plt.ylabel('F1 Score')
    plt.xlim(0,15)
    plt.legend()

    # Add thresholds above the two bar graphs
    if thresholds:
        for idx,rects in enumerate(zip(bar1,bar2)):
            higher=np.argmax([rects[0].get_height(),rects[1].get_height()])
            rect=rects[higher]
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()-(higher/2.), height, '%.2f' % thresholds[idx], ha='center', va='bottom')


# In[8]:


hypers = np.random.choice(np.arange(1,20),(10,2),replace=True)
for hyp in hypers:
    print(hyp)

