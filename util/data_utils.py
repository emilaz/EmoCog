import pandas as pd
import numpy as np
import os
import ast
import pickle
import glob


def generate_filename(configs, cut=True):
    """
    Generates a filename out of the configs
    Input: Configs
    Output: Filename
    """
    fname = ''
    for k, v in configs.items():
        if k == 'cutoff':
            if not cut:
                continue
        fname += k + "_" + str(v) + '_'
    return fname[:-1]


def generate_configs_from_file(filename, cutoff=None):
    """
    Generates configs from a filename
    Input: Filename
    Output: Configs
    """
    split_list = np.array(filename.split('_'))
    configs = dict(zip(split_list[::2], split_list[1::2]))
    configs['days'] = ast.literal_eval(configs['days'])
    configs['wsize'] = int(configs['wsize'])
    configs['ratio'] = float(configs['ratio'])
    try:
        configs['sliding'] = int(configs['sliding'])
    except ValueError:
        configs['sliding'] = False
    configs['expvar'] = int(configs['expvar'])
    try:
        configs['cutoff'] = float(configs['cutoff'])
    except KeyError:
        None
    try:
        if configs['shuffle'] == 'True':
            configs['shuffle'] = True
        else:
            configs['shuffle'] = False
    except KeyError:
        None
    if cutoff is not None:
        configs['cutoff'] = cutoff
    return configs


def save_data_to_file(x, y, x_ev, y_ev, configs):
    """
    This method saves the generated train and test data to a file, given the corresponding configs. Automatically generates an appropriate filename.
    Input: Train and test data, configs
    Output: None
    """
    fname = generate_filename(configs, False)
    link = os.path.join('/home/emil/EmoCog/data/new_labels/train_test_datasets',configs['patient'], fname)
    # save stuff to file:
    df = pd.DataFrame(data=[[x, y, x_ev, y_ev]], columns=['x_tr', 'y_tr', 'x_ev', 'y_ev'])
    df.to_hdf(link, key='df')


def load_data_from_file(configs):
    """
    This method loads the data (train and test) into memory, given the configs. Obviously only works when data for these configs has been generated and saved some previous time
    Input: Configs
    Output: Train and test data according to these configs
    """
    fname = generate_filename(configs, False)
    link = os.path.join('/home/emil/EmoCog/data/new_labels/train_test_datasets',configs['patient'], fname)
    print(link)
    df = pd.read_hdf(link)
    x = df['x_tr'][0]
    y = df['y_tr'][0]
    x_ev = df['x_ev'][0]
    y_ev = df['y_ev'][0]
    return x, y, x_ev, y_ev


def load_data_from_path(path):
    """
    This methods loads data given a file path
    Input: File path
    Output: x,y,x_ev,y_ev
    """
    df = pd.read_hdf(path)
    x = df['x_tr'][0]
    y = df['y_tr'][0]
    x_ev = df['x_ev'][0]
    y_ev = df['y_ev'][0]
    return x, y, x_ev, y_ev


def save_results(df, configs, methodtype):
    """
    This method saves the obtained results to a file for later inspection.
    Input: Datafrme with results, config file, ML approach used
    Output: None
    """
    fname = generate_filename(configs)
    path = os.path.join('/home/emil/EmoCog/data/new_labels/results/', configs['patient'], methodtype)
    if not os.path.exists(path):
        os.makedirs(path)
    link = os.path.join('/home/emil/EmoCog/data/new_labels/results/', configs['patient'], methodtype, fname)
    df.to_hdf(link, key='df')


def load_results(configs, methodtype):
    """
    Returns a dataframe with results, given the configs and the ML-method type in question
    Input: Configs, Method
    Output: Dataframe with results
    """
    fname = generate_filename(configs)
    link = os.path.join('/home/emil/EmoCog/data/new_labels/results', configs['patient'], methodtype, fname)
    df = pd.read_hdf(link)
    return df


def save_processing_tools(pca, artifact_paras, standard_paras, good_chans, configs):
    """
    This method saves the obtained PCA to a file for later inspection, as well as
    the parameters needed for artifact removal and standardization and good channels
    Input: classifier, config file
    Output: None
    """
    fname = generate_filename(configs) + '.pkl'
    if not os.path.exists(os.path.join('/home/emil/EmoCog/data/new_labels/pca_models', configs['patient'])):
        os.mkdir(os.path.join('/home/emil/EmoCog/data/new_labels/pca_models', configs['patient']))
    link = os.path.join('/home/emil/EmoCog/data/new_labels/pca_models', configs['patient'], configs['patient'], fname)
    tools = {'Model': pca, 'Artifact Parameter': artifact_paras, 'Standardization Parameter': standard_paras,
             'GoodChans': good_chans}
    with open(link, "wb") as f:
        pickle.dump(tools, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_processing_tools(configs):
    """
    Returns the trained PCA and artifact/standardization parameters given the configs
    Input: Configs
    Output: Trained PCA model, artifact parameter, standardization parameter
    """
    fname = generate_filename(configs, False) + '.pkl'
    link = os.path.join('/home/emil/EmoCog/data/new_labels/pca_models', configs['patient'], fname)
    with open(link, "rb") as f:
        tools = pickle.load(f)
    return tools


def load_processing_tools_from_path(path):
    """
    Returns the trained PCA and artifact/standardization parameters given the configs
    Input: Path
    Output: Trained PCA model, artifact parameter, standardization parameter
    """
    with open(path, "rb") as f:
        tools = pickle.load(f)
    return tools


def save_classifier(classifier, best_thr, configs, methodtype):
    """
    This method saves the trained classifier to a file for later use.
    Input: Classifier, config file, ML approach used
    Output: None
    """
    fname = generate_filename(configs) + '.pkl'
    if not os.path.exists(os.path.join('/home/emil/EmoCog/data/new_labels/classifier', configs['patient'], methodtype)):
        os.makedirs(os.path.join('/home/emil/EmoCog/data/new_labels/classifier', configs['patient'], methodtype))
    link = os.path.join('/home/emil/EmoCog/data/new_labels/classifier', configs['patient'], methodtype, fname)
    classifier_stuff = {'classifier': classifier, 'threshold': best_thr}
    with open(link, "wb") as f:
        pickle.dump(classifier_stuff, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_classifier(configs, methodtype):
    """
    Returns the trained classifier, given the configs and the ML-method type in question
    Input: Configs, Method
    Output: Classifier with results
    """
    fname = generate_filename(configs) + '.pkl'
    link = os.path.join('/home/emil/EmoCog/data/new_labels/classifier', configs['patient'], methodtype, fname)
    with open(link, "rb") as f:
        classifier_stuff = pickle.load(f)
    return classifier_stuff['classifier'], classifier_stuff['threshold']


def generate_graph_link(configs):
    # first, create path
    root = '/home/emil/EmoCog/data/new_labels/images'
    folder = str(configs['patient']) + '_' + str(configs['days'])
    path = os.path.join(root, folder)
    if not os.path.exists(path):
        os.mkdir(path)
    # then, create filename
    new_dict = dict()
    for k,v in configs.items():
        if k != 'patient' and k != 'days':
            new_dict[k] = v
    fil = generate_filename(new_dict)
    return os.path.join(path, fil)


def remove_temporary_data():
    print('Deleting temporary data.')
    temporary_files = glob.glob('/home/emil/EmoCog/data/temporary/*')
    [os.remove(t) for t in temporary_files]