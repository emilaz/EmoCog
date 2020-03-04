
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider


def plot_raw_data(data,chans=None,bad_coords= []):
    if chans is None:
        chans = range(data.shape[0])
    fig, ax = plt.subplots(figsize=(10,10))
    plt.subplots_adjust(bottom=0.25)
    for ch in chans:
        plt.plot(data[ch])
    plt.axis([0, 100000, -1000, 1000])
    for c in bad_coords:
        ax.axvspan(c[0],c[1],color='red',alpha=.5)
    axcolor = 'lightgoldenrodyellow'
    axpos = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
    spos = Slider(axpos, 'Pos', 0.1, len(data[0]))
    def update(val): #needed for slider function of plot_raw_data
        pos = spos.val
        ax.axis([pos,pos+50000,-500,500])
        fig.canvas.draw_idle()
    spos.on_changed(update)
    plt.show();

    
def plot_features(data):
    plts = data.shape[0]//20 +1 #we want 20 per plot
    xsize=10
    ysize=5
    fig=plt.figure()
    for k in range (0,plts):
        ax=fig.add_subplot(xsize,ysize,k+1)
        l = ax.plot(data[k*20:(k+1)*20])
        plt.axis([0, 1000, 0, 10])
        sframe = Slider(fig.add_subplot(50,1,50), 's', 0, len(data[0])-1, valinit=0)
        def update(val):
            frame = np.around(sframe.val)
            #l.set_data(readlist[k][frame,:,:])
            ax.axis([pos,pos+1000,0,10])
    sframe.on_changed(update)
    plt.show()


def plot_pc(pca,data):
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


