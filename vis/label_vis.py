
import numpy as np
import matplotlib.pyplot as plt


# #plot the nan ratio
def plot_nan_ratio(all_video_preds):  
    br=np.unique(np.array(all_video_preds, dtype='float'), return_counts=True) #check these elements
    sum_nans=np.sum(br[1][2:]) #has to be done separately, since nans are counted individually here
    vals=([str(br[0][0]),str(br[0][1]),str(br[0][2])],[br[1][0],br[1][1],sum_nans])
    print(vals[0],vals[1])
    plt.bar(vals[0],vals[1])
    plt.title("Occurences of 'Happy'/'Not Happy'/'N/A' predictions (total of %d samples)" % (len(all_video_preds)))
    plt.xlabel('Prediction')
    plt.ylabel('Occurences')
    plt.show()


def plot_happy_ratio(regression_labels, regression_labels_nan_fraction='b'): #this plots the happy/non-happy per label. if available, also plots a heatmap of the nan-ratio per each
    plt.figure(figsize=(15,5))
    plt.scatter(range(len(regression_labels)),regression_labels, c=regression_labels_nan_fraction, s=2)
    plt.title('Mean Happiness')
    plt.ylabel('Value')
    plt.xlabel('Data point no.')
    if regression_labels_nan_fraction is not 'b':
        cbar=plt.colorbar()
        cbar.set_label('Ratio Pred:NaN')
    plt.show()

def plot_happy_bars(y,y_ev):
    entries, counts = np.unique(y, return_counts = True)
    entries, counts_ev = np.unique(y_ev, return_counts = True)
    happies = [counts[1],counts_ev[1]]
    nohappies  = [counts[0],counts_ev[0]]
    bar_width = 0.35
    plt.xticks(np.arange(0,2)+bar_width/2,['Train','Test'])
    plt.bar(np.arange(2), happies, bar_width, label='Happy',color='blue')
    plt.bar(np.arange(2)+bar_width, nohappies,bar_width, label='Not Happy',color='red')
    plt.legend()
    plt.ylabel('Counts')
    plt.title('Number of Not Happy/Happy labels')
    plt.show()

