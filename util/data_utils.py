import pandas as pd
import numpy as np
import os
import ast
import pickle

"""
Generates a filename out of the configs
Input: Configs
Output: Filename
"""


def generate_filename(configs,cut=True):
    fname = ''
    for k,v in configs.items():
        if k == 'cutoff':
            if not cut:
                continue
        fname += k+'_'+str(v)+'_'
    return fname[:-1]

"""
Generates configs from a filename
Input: Filename
Output: Configs
"""
def generate_configs_from_file(filename,cutoff=None):
    split_list = np.array(filename.split('_'))
    configs = dict(zip(split_list[::2], split_list[1::2]))
    configs['days'] = ast.literal_eval(configs['days'])
    configs['wsize'] = int(configs['wsize'])
    configs['ratio']= float(configs['ratio'])
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
    

"""
This method saves the generated train and test data to a file, given the corresponding configs. Automatically generates an appropriate filename.
Input: Train and test data, configs
Output: None
"""
def save_data_to_file(x, y, x_ev, y_ev, configs):
    fname = generate_filename(configs, False)
    link = os.path.join('/home/emil/EmoCog/data/new_labels/train_test_datasets',fname)
    #save stuff to file:
    df = pd.DataFrame(data=[[x, y, x_ev, y_ev]],columns=['x_tr','y_tr','x_ev','y_ev'])
    df.to_hdf(link,key='df')
    
    
"""
This method loads the data (train and test) into memory, given the configs. Obviously only works when data for these configs has been generated and saved some previous time
Input: Configs
Output: Train and test data according to these configs
"""
def load_data_from_file(configs):
    fname = generate_filename(configs,False)
    link = os.path.join('/home/emil/EmoCog/data/new_labels/train_test_datasets',fname)
    print(link)
    df = pd.read_hdf(link)
    x = df['x_tr'][0]
    y = df['y_tr'][0]
    x_ev = df['x_ev'][0]
    y_ev = df['y_ev'][0]
    return x,y,x_ev,y_ev



"""
This methods loads data given a file path
Input: File path
Output: x,y,x_ev,y_ev
"""
def load_data_from_path(path):
    df = pd.read_hdf(path)
    x = df['x_tr'][0]
    y = df['y_tr'][0]
    x_ev = df['x_ev'][0]
    y_ev = df['y_ev'][0]
    return x,y,x_ev,y_ev


"""
This method saves the obtained results to a file for later inspection.
Input: Datafrme with results, config file, ML approach used
Output: None
"""
def save_results(df, configs,methodtype):
    fname = generate_filename(configs)
    if not os.path.exists(os.path.join('/home/emil/EmoCog/data/new_labels/results/',methodtype)):
        os.mkdir('/home/emil/EmoCog/data/new_labels/results/'+methodtype)
    link = os.path.join('/home/emil/EmoCog/data/new_labels/results/',methodtype,fname)
    df.to_hdf(link,key='df')

    
"""
Returns a dataframe with results, given the configs and the ML-method type in question
Input: Configs, Method
Output: Dataframe with results
"""
def load_results(configs,methodtype):
    fname = generate_filename(configs)
    link = os.path.join('/home/emil/EmoCog/data/new_labels/results',methodtype,fname)
    df = pd.read_hdf(link)
    return df


"""
This method saves the obtained PCA to a file for later inspection, as well as the parameters needed for artifact removal and standardization
Input: classifier, config file
Output: None
"""
def save_processing_tools(pca, artifact_paras, standard_paras, good_chans, configs):
    fname = generate_filename(configs)+'.pkl'
    if not os.path.exists('/home/emil/EmoCog/data/new_labels/pca_models'):
        os.mkdir('/home/emil/EmoCog/data/new_labels/pca_models')
    link = os.path.join('/home/emil/EmoCog/data/new_labels/pca_models',fname)
    tools ={'Model':pca, 'Artifact Parameter':artifact_paras, 'Standardization Parameter':standard_paras, 'GoodChans':good_chans}
    with open(link, "wb") as f:
         pickle.dump(tools, f, protocol=pickle.HIGHEST_PROTOCOL)

                
"""
Returns the trained PCA and artifact/standardization parameters given the configs 
Input: Configs, Method
Output: Trained PCA model, artifact parameter, standardization parameter
"""
def load_processing_tools(configs):
    fname = generate_filename(configs,False)+'.pkl'
    link = os.path.join('/home/emil/EmoCog/data/new_labels/pca_models',fname)
    with open(link, "rb") as f:
         tools = pickle.load(f)
    return tools


"""
This method saves the trained classifier to a file for later use.
Input: Classifier, config file, ML approach used
Output: None
"""
def save_classifier(classifier, best_thr, configs, methodtype):
    fname = generate_filename(configs)+'.pkl'
    if not os.path.exists(os.path.join('/home/emil/EmoCog/data/new_labels/classifier',methodtype)):
        os.mkdir(os.path.join('/home/emil/EmoCog/data/new_labels/classifier',methodtype))
    link = os.path.join('/home/emil/EmoCog/data/new_labels/classifier',methodtype,fname)
    classifier_stuff = {'classifier':classifier, 'threshold':best_thr}
    with open(link, "wb") as f:
         pickle.dump(classifier_stuff, f, protocol=pickle.HIGHEST_PROTOCOL)
    
"""
Returns the trained classifier, given the configs and the ML-method type in question
Input: Configs, Method
Output: Classifier with results
"""
def load_classifier(configs,methodtype):
    fname = generate_filename(configs)+'.pkl'
    link = os.path.join('/home/emil/EmoCog/data/new_labels/classifier',methodtype,fname)
    with open(link, "rb") as f:
         classifier_stuff = pickle.load(f)
    return classifier_stuff['classifier'], classifier_stuff['threshold']
