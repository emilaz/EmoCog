
# coding: utf-8

# In[ ]:


import supereeg as se
import numpy as np
import pandas as pd
from nilearn import plotting as ni_plt
import pdb

def plot_ecog_electrodes_mni_from_file(mni_coords_fullfile,chan_num_min=-1,chan_num_max=-1,num_grid_chans=64,colors=list()):
    'Plots ECoG electrodes from MNI coordinate file'
    #Example code to run it: 
         #import sys
         #sys.path.append('/home/stepeter/AJILE/stepeter_sandbox/ECoG_Preprocessing')
         #from plot_ecog_electrodes_mni import *

         #mni_coords_fullfile='/data2/users/stepeter/mni_coords/a0f66459/294e1c_Trodes_MNIcoords.txt'
         #plot_ecog_electrodes_mni(mni_coords_fullfile,chan_num_min=-1,chan_num_max=-1,num_grid_chans=64)
        
    #NOTE: A warning may pop up the first time running it, leading to no output. Rerun this function, and the plots should appear.
    
    #Load in MNI file
    mni_file = np.loadtxt(mni_coords_fullfile, delimiter=",")
    
    #Specify which channels to plot (from min to max number)
    if chan_num_min==-1:
        chan_num_min=0
    if chan_num_max==-1:
        chan_num_max=mni_file.shape[0]-1
    slice_inds=slice(chan_num_min,chan_num_max+1)
    
    #Create dataframe for electrode locations
    locs=pd.DataFrame({'x': mni_file[slice_inds,0].T,
                      'y': mni_file[slice_inds,1].T,
                      'z': mni_file[slice_inds,2].T})
    
    #Label strips/depths differently for easier visualization (or use defined color list)
    if len(colors)==0:
        for s in range(locs.shape[0]):
            if s>=num_grid_chans:
                colors.append('r')
            else:
                colors.append('b')
        
    #Plot the result
    ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                           node_kwargs={'alpha': 0.5, 'edgecolors': None},
                           node_size=10, node_color=colors)

def plot_ecog_electrodes_mni_from_file_and_labels(mni_coords_fullfile,chan_labels,num_grid_chans=64,colors=list()):
    'Plots ECoG electrodes from MNI coordinate file'
    #Example code to run it: 
         #import sys
         #sys.path.append('/home/stepeter/AJILE/stepeter_sandbox/ECoG_Preprocessing')
         #from plot_ecog_electrodes_mni import *

         #mni_coords_fullfile='/data2/users/stepeter/mni_coords/a0f66459/a0f66459_MNI_atlasRegions.xlsx'
         #plot_ecog_electrodes_mni_from_file_and_labels(mni_coords_fullfile,chan_num_min=-1,chan_num_max=-1,num_grid_chans=64)
        
    #NOTE: A warning may pop up the first time running it, leading to no output. Rerun this function, and the plots should appear.
    
    #Load in MNI file
    mni_file = pd.read_excel(mni_coords_fullfile, delimiter=",")
    
    
    #Create dataframe for electrode locations
    locs=mni_file.loc[mni_file['Electrode'].isin(chan_labels)][['X coor', 'Y coor', 'Z coor']]

    
    #Label strips/depths differently for easier visualization (or use defined color list)
    if len(colors)==0:
        for s in range(locs.shape[0]):
            if s>=num_grid_chans:
                colors.append('r')
            else:
                colors.append('b')
        
    #Plot the result
    ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                           node_kwargs={'alpha': 0.5, 'edgecolors': None},
                           node_size=10, node_color=colors)
    
def plot_ecog_electrodes_mni_direct(mni_coords,num_grid_chans=64,colors=list()):
    'Plots ECoG electrodes from MNI coordinate file'
    #Example code to run it: 
        # import sys
        # sys.path.append('/home/stepeter/AJILE/stepeter_sandbox/ECoG_Preprocessing')
        # from plot_ecog_electrodes_mni import *

        # colors=list()
        # cmap = matplotlib.cm.get_cmap('jet')
        # for i in range(virtualCoords_pos.shape[1]):
        # colors.append(np.asarray(cmap(frac_sbj_virtual_grid[0,i]))[0:3])
        # colors = np.asarray(colors)
        # colors = list(map(lambda x: x[0], np.array_split(colors, colors.shape[0], axis=0)))
        # plot_ecog_electrodes_mni_direct(virtualCoords_pos.T,num_grid_chans=64,colors=colors)
        
    #NOTE: If running in Jupyter, use '%matplotlib inline' instead of '%matplotlib notebook'
    
    #Create dataframe for electrode locations
    locs=pd.DataFrame(mni_file.loc[mni_file['Electrode'].isin(chan_labels)]['Coordinates'])
    #locs=pd.DataFrame({'x': mni_coords[:,0].T,
    #                  'y': mni_coords[:,1].T,
    #                  'z': mni_coords[:,2].T})
    
    #Label strips/depths differently for easier visualization (or use defined color list)
    if len(colors)==0:
        for s in range(locs.shape[0]):
            if s>=num_grid_chans:
                colors.append('r')
            else:
                colors.append('b')
   
    #Plot the result
    ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                           node_kwargs={'alpha': 0.5, 'edgecolors': None},
                           node_size=10, node_color=colors)


# In[ ]:


# import sys
# sys.path.append('/home/stepeter/AJILE/stepeter_sandbox/ECoG_Preprocessing')

# mni_coords_fullfile='/data2/users/stepeter/mni_coords/a0f66459/a0f66459_MNI_atlasRegions.xlsx'
# plot_ecog_electrodes_mni_from_file_and_labels(mni_coords_fullfile,chan_labels=['GRID1', 'GRID2'], num_grid_chans=64)

