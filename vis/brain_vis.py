import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
from nilearn import plotting as ni_plt
import matplotlib.gridspec as gridspec
import sys

sys.path.append('..')
from util.analysis_utils import get_important_electrodes_bins_goodchans
from util.data_utils import generate_graph_link


def plot_all(link_to_ecog_file, configs, top_n=5):
    """
    Function that plots the 10 most important electrodes on the brain & most important frequency bin
    :param link_to_ecog_file: we need the .h5 file containing the xyz coordinates
    :param configs: configs of case we want to plot
    :return:
    """
    imp_electrodes, imp_bins, good_chans = get_important_electrodes_bins_goodchans(configs)
    title = generate_graph_link(configs)
    plot_important_frequencies(imp_bins, title)
    plot_important_electrodes(imp_electrodes, good_chans, title)
    best_elec_ind = imp_electrodes.argsort()[::-1][:top_n]
    best_elec = imp_electrodes.copy()
    best_elec[:] = 1
    best_elec[best_elec_ind] = 0
    plot_brain_new(h5_fn=link_to_ecog_file, chan_labels=good_chans, colors=best_elec, title=title)


def plot_important_frequencies(imp_bins, title):
    plt.figure()
    df = pd.DataFrame(
        {'Frequencies': ['0-1', '2', '3-4', '5-8', '9-16', '17-32', '33-64', '65-150'], 'Importances': imp_bins})
    sns.barplot(data=df, x='Frequencies', y='Importances')
    plt.ylabel('Importance')
    plt.title('Normalized Importance of Each Frequency Bin')
    plt.savefig(title + ' Freqs.png')
    plt.show()
    plt.close()


def plot_important_electrodes(imp_electrodes, good_chans, title):
    plt.figure()
    imp_ele = np.argsort(imp_electrodes)[-5:]
    imp = np.zeros(len(good_chans), dtype='bool')
    imp[imp_ele] = 1
    df = pd.DataFrame({'Labels': good_chans, 'Importances': imp_electrodes, 'Top 5': imp})
    plt.figure(figsize=(10, 20))
    sns.barplot(data=df, y='Labels', x='Importances', hue='Top 5')
    plt.title('Normalized Importance of Each Electrode')
    plt.savefig(title + ' Electrodes.png')
    plt.show()
    plt.close()


def plot_brain_new(h5_fn=None, chan_labels='all', num_grid_chans=64, colors=None, node_size=50,
                   figsize=(16, 6), sides_2_display='auto', node_edge_colors=None,
                   alpha=0.5, edge_linewidths=3, ax_in=None,
                   allLH=False, title=None):
    """
    Plots ECoG electrodes from MNI coordinate file (only for specified labels)
    
    Example code to run it: 
         import sys
         sys.path.append('/home/stepeter/AJILE/stepeter_sandbox/ECoG_Preprocessing')
         from Steve_libPreprocess import *
         subjID = 'a0f66459'
         day = '3'
         h5_fn='/nas/ecog_project/derived/processed_ecog/'+subjID+'/full_day_ecog/'+subjID+'_fullday_'+day+'.h5'
         chan_labels=list(['GRID1','GRID2']) #or chan_labels='all'
         plot_ecog_electrodes_mni_from_file_and_labels(h5_fn,chan_labels,num_grid_chans=64)
        
    NOTE: If running in Jupyter, use '%matplotlib inline' instead of '%matplotlib notebook'
    """

    # h5_fn = '/nas/ecog_project/derived/processed_ecog/cb46fd46/full_day_ecog/cb46fd46_fullday_3.h5'
    # Load channel locations
    chan_info = pd.read_hdf(h5_fn, key='chan_info', mode='r')
    # change label names to conform to our allupper convention
    chan_info.columns = [f.upper() for f in chan_info.columns]
    #### JUST FOR NOW
    # colors = [1 for r in chan_info.columns]
    # colors = ['LTO' in r or 'LO' in r or 'LOF' in r for r in chan_info.columns]
    # Create dataframe for electrode locations
    if chan_labels is 'all':
        locs = chan_info.loc[['X', 'Y', 'Z'], :].transpose()
    else:
        locs = chan_info.loc[['X', 'Y', 'Z'], chan_labels].transpose()
    locs.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)
    # filter out
    chan_loc_x = locs['x'].values

    # Remove NaN electrode locations (no location info)
    nan_drop_inds = np.nonzero(np.isnan(chan_loc_x))[0]
    locs.dropna(axis=0, inplace=True)  # remove NaN locations

    colors = list(np.delete(colors, nan_drop_inds))

    # Decide whether to plot L or R hemisphere based on x coordinates
    if len(sides_2_display) > 1:
        N, fig, axes, sides_2_display = _setup_subplot_view(locs, sides_2_display, figsize)
    else:
        N = 1
        axes = ax_in
        if allLH:
            average_xpos_sign = np.mean(np.asarray(locs['x']))
            if average_xpos_sign > 0:
                locs['x'] = -locs['x']
            sides_2_display = 'l'

    # Rearrange to plot non-grid electrode first
    if num_grid_chans > 0:  # isinstance(colors, list):
        locs2 = locs.copy()
        locs2['x'] = np.concatenate((locs['x'][num_grid_chans:], locs['x'][:num_grid_chans]), axis=0)
        locs2['y'] = np.concatenate((locs['y'][num_grid_chans:], locs['y'][:num_grid_chans]), axis=0)
        locs2['z'] = np.concatenate((locs['z'][num_grid_chans:], locs['z'][:num_grid_chans]), axis=0)

        if isinstance(colors, list):
            colors2 = colors.copy()
            colors2 = colors[num_grid_chans:] + colors[:num_grid_chans]
        elif isinstance(colors, np.ndarray):
            colors2 = np.append(colors[num_grid_chans:], colors[:num_grid_chans])
        else:
            colors2 = colors
    else:
        locs2 = locs.copy()
        if isinstance(colors, list) or isinstance(colors, np.ndarray):
            colors2 = colors.copy()
        else:
            colors2 = colors  # [colors for i in range(locs2.shape[0])]

    # Plot the result
    _plot_electrodes(locs2, node_size, colors2, axes, sides_2_display, N, node_edge_colors, alpha, edge_linewidths)
    # add_colorbar(fig, min(colors2), max(colors2), 'viridis', label_name='')
    # fig.suptitle('Normalized Importance of Each Electrode Mapped to Brain', fontsize=14)
    plt.savefig(title + '_brain.eps', format='eps')
    # plt.savefig('justthebrain.png')
    plt.show()
    plt.close()


def _setup_subplot_view(locs, sides_2_display, figsize):
    """
    Decide whether to plot L or R hemisphere based on x coordinates
    """
    if sides_2_display == 'auto':
        average_xpos_sign = np.mean(np.asarray(locs['x']))
        if average_xpos_sign > 0:
            sides_2_display = 'yrz'
        else:
            sides_2_display = 'ylz'

    # Create figure and axes
    if sides_2_display == 'ortho':
        N = 1
    else:
        N = len(sides_2_display)

    if sides_2_display == 'yrz' or sides_2_display == 'ylz':
        gridspec.GridSpec(0, 3)
        fig, axes = plt.subplots(1, N, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, N, figsize=figsize)
    return N, fig, axes, sides_2_display


def _plot_electrodes(locs, node_size, colors, axes, sides_2_display, N, node_edge_colors, alpha, edge_linewidths):
    """
    Handles plotting
    """
    if N == 1:
        ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                               node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,
                                            'linewidths': edge_linewidths},
                               node_size=node_size, node_color=colors, axes=axes, display_mode=sides_2_display)
    elif sides_2_display == 'yrz' or sides_2_display == 'ylz':
        colspans = [5, 6, 5]  # different sized subplot to make saggital view similar size to other two slices
        current_col = 0
        total_colspans = int(np.sum(np.asarray(colspans)))
        for ind, colspan in enumerate(colspans):
            axes[ind] = plt.subplot2grid((1, total_colspans), (0, current_col), colspan=colspan, rowspan=1)
            ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                                   node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,
                                                'linewidths': edge_linewidths,
                                                # 'cmap': 'Paired'},
                                                'cmap': 'tab20c'},
                                   node_size=node_size, node_color=colors, axes=axes[ind],
                                   display_mode=sides_2_display[ind])
            current_col += colspan
    else:
        for i in range(N):
            ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                                   node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,
                                                'linewidths': edge_linewidths},
                                   node_size=node_size, node_color=colors, axes=axes[i],
                                   display_mode=sides_2_display[i])


def add_colorbar(f_in, vmin, vmax, cmap, width=0.025, height=0.16, horiz_pos=.91, border_width=1.5,
                 tick_len=0, adjust_subplots_right=0.84, label_name='', tick_fontsize=12,
                 label_fontsize=12, label_pad=15, label_y=0.6, label_rotation=0, fontweight='normal',
                 fontname='DejaVu Sans'):
    '''
    Adds colorbar to existing plot based on vmin, vmax, and cmap
    '''
    f12636, a14u3u43 = plt.subplots(1, 1, figsize=(0, 0))
    im = a14u3u43.imshow(np.random.random((10, 10)), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.close(f12636)
    f_in.subplots_adjust(right=adjust_subplots_right)
    vert_pos = (1 - height) / 2
    cbar_ax = f_in.add_axes([horiz_pos, vert_pos, width, height])
    cbar = f_in.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([vmin, 0, vmax])
    cbar.ax.set_yticklabels(['{:.2f}'.format(vmin), 0, '{:.2f}'.format(vmax)], fontsize=tick_fontsize,
                            weight=fontweight, fontname=fontname)
    cbar.ax.tick_params(length=tick_len)
    cbar.outline.set_linewidth(border_width)
    cbar.set_label(label_name, rotation=label_rotation, fontsize=label_fontsize,
                   weight=fontweight, labelpad=label_pad, y=label_y, fontname=fontname)


if __name__ == '__main__':
    # patient = ['cb46fd46']
    # days = [[3,4,5,6,7]]
    # wsize = 100
    # sliding = 25
    # shuffle = False
    # expvar = 90
    # ratio = .8
    # configs = dict()
    # configs['patient'] = patient
    # configs['days'] = days
    # configs['wsize'] = wsize
    # configs['sliding'] = sliding
    # configs['expvar'] = expvar
    # configs['ratio'] = ratio
    # configs['shuffle'] = shuffle
    # configs['cutoff'] = .3
    # print('los', configs)
    h5_fn = '/nas/ecog_project/derived/processed_ecog/e5bad52f/full_day_ecog/e5bad52f_fullday_3.h5'
    # plot_all(h5_fn, configs)
    plot_brain_new(h5_fn)
    # plot_ecog_electrodes_mni_from_file_and_labels(h5_fn)
