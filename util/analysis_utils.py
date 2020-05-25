import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import util.data_utils as dutil


def get_important_electrodes_bins_goodchans(configs):
    # first, we need to know which principal directions were important for the RF classifier. So, load classifier
    classifier_stuff = dutil.load_classifier(configs,'RF')
    pca_stuff = dutil.load_processing_tools(configs)

    good_chans = pca_stuff['GoodChans']
    pca = pca_stuff['Model']  # this is our trained PCA
    classifier = classifier_stuff[0]  # this is our trained Random Forest classifier

    importances = classifier.feature_importances_
    loadings = abs(pca.components_)

    # each feature has a certain importance (from RF).
    # These features are derived from PCA, which compressed from 8 PSD bins per electrode to N principal comps
    # loadings are a (PCs x (8xElectrodes)) matrix. Multiply each PC row with importance of that PC
    # to get weighted importance of each pre-PCA feature for each principal component
    weighted_loadings = importances[:,None] * loadings

    # sum the weighted importances over all PC's to get a single number for each pre-PCA feature
    sum_loadings = np.sum(weighted_loadings,axis = 0)

    # now, get the contribution of each electrode. take into account that each electrode has 8 bins
    imp_electrodes = np.sum(sum_loadings.reshape(8,-1),axis=0)
    # then, get the contribution of each frequency bin. take
    imp_bin = np.sum(sum_loadings.reshape(8,-1),axis=1)
    return imp_electrodes, imp_bin, good_chans

