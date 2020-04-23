""""
This class takes the OpenFace output for a given patient, adds time and day as columns, saves per day as dataframe
"""""

import pandas as pd
import gc

import os
import sys
import functools
import glob
from datetime import time
from multiprocessing import Pool
import dask.dataframe as dd


def fill_row(times,index,row):
    #create the filename from line
    pat = row['patient']
    sess = row['session']
    vid = row['video']
    fname = '_'.join([pat,sess,vid])
    fname = '.'.join([fname,'avi'])
    #find the file in the videotime infostuff
    line = times[times['filename']==fname]
    # #make frames start at 0 instead of 1
    # row['frame']=row['frame']-1
    # if row['frame']<0:
    #     raise ValueError('Somehow the frame number here became negative. Check what happened.')
    #crate time object
    vid_time = time(line['hour'].iloc[0],line['minute'].iloc[0],line['second'].iloc[0],line['microsecond'].iloc[0])
    #add column wih the important info
    return [*row,vid_time,line['merge_day'].iloc[0]]


def load_patient(patient_dir):
    try:
        curr_df = pd.read_hdf(os.path.join(patient_dir, 'hdfs', 'Happy_predictions.hdf'), '/data')
        # curr_df = dd.read_hdf(os.path.join(patient_dir, 'hdfs', 'Happy_predictions.hdf'), '/data')
        if 'timestamp' in curr_df.columns:
            del curr_df['timestamp']

        if len(curr_df) and 'Happy' in curr_df.columns:
            return curr_df

    except AttributeError as e:
        print(e)
    except ValueError as e:
        print(e)
    except KeyError as e:
        print(e)



if __name__=='__main__':
    pat = sys.argv[sys.argv.index('-patient')+1]
    dir = sys.argv[sys.argv.index('-dir')+1]
    #where are the files for this patient?
    os.chdir(dir)
    #first, read in the data we have so far
    PATIENT_DIRS = [
        x for x in glob.glob('*cropped') if 'hdfs' in os.listdir(x)
        and pat in x and 'Happy_predictions.hdf' in os.listdir(os.path.join(x,'hdfs'))
    ]

    # this gets the realtimes for the videos
    time_path = os.path.join('/data1/ecog_project/derived/processed_ecog/',pat,'full_day_ecog/vid_start_end_merge.csv')
    times = pd.read_csv(time_path)

    # load all files of this patient into memory using multiprocessing
    pool = Pool(8)
    ret = pool.map(load_patient,PATIENT_DIRS)
    print('collected data, now concatenating..')
    df = pd.concat(ret)
    print('concatenated.')
    gc.collect()
    all_rows = df.iterrows()
    # combined = [(idx, row, times) for (idx, row) in all_rows]
    print('time to add times')
    party = functools.partial(fill_row,times)
    pool = Pool(8)
    yass = pool.starmap(party, all_rows)
    cols = [*df.columns, 'steve_time', 'merge_day']
    del df
    gc.collect()
    print('now make this a dataframe')
    new_df = pd.DataFrame(yass, columns=cols)
    del yass
    gc.collect()
    # new_df = pd.DataFrame(pool.starmap(fill_row, combined), columns=[*df.columns, 'steve_time', 'merge_day'])
    print('done, now lets save to file.')
    # sort by day, save by day.
    for day in pd.unique(new_df['merge_day']):
        print('day',day)
        curr_df = new_df[new_df['merge_day'] == day]
        sorted_curr_df = curr_df.sort_values(['steve_time','frame']).reset_index(drop=True)
        curr_path = os.path.join('/home/emil/data/hdf_data/by_day_new',pat)
        if not os.path.exists(curr_path):
            os.mkdir(curr_path)
        sorted_curr_df.to_hdf(os.path.join(curr_path,pat+'_day_'+ str(day) + '.hdf'), key='df')



