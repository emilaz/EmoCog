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
    pca = pca_stuff['Model']
    classifier = classifier_stuff[0]

    importances = classifier.feature_importances_
    loadings = abs(pca.components_)

    # ok now how does this work? first, which components had importance at all, and how much?
    weighted_loadings = importances[:,None] * loadings

    #sum the contribution*importance over all PC's
    sum_loadings = np.sum(weighted_loadings,axis = 0)

    #now, get the contribution of each electrode. take into account that each electrode has 8 bins
    imp_electrodes = np.sum(sum_loadings.reshape(8,-1),axis=0)
    #then, get the contribution of each frequency bin. take
    imp_bin = np.sum(sum_loadings.reshape(8,-1),axis=1)
    return imp_electrodes, imp_bin, good_chans

