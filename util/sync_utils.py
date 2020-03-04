
import numpy as np
import pandas as pd
import os
from datetime import datetime, time

"""
Given a day, finds the corresponding video file I guess
"""
#TODO: Check if session is longer than a day
def find_paths(patient_name,day_no):
    #create filename:
    filename = patient_name+'_fullday_'+str(day_no)+'.h5'
    video_filename = patient_name+'_day_'+str(day_no)+'.hdf'
    path = os.path.join('/nas/ecog_project/derived/processed_ecog',patient_name,'full_day_ecog')
    ecog_path = os.path.join(path, filename)
    video_path = os.path.join('/home/emil/data/hdf_data/by_day/',video_filename)
#     print(video_path)
#     print(ecog_path)
    #check if file exists
    if not (os.path.exists(ecog_path) and os.path.exists(video_path)):
        FileNotFoundError('Files for day {} do not exist. Check if day or patient name is wrong.'.format(day_no))
        return
    return ecog_path, video_path

"""
This function finds out how many seconds into the day the given session start
Input: Path to video session hdf file
Output: Seconds passed since midnight until video session began"""
def find_start_and_end_time(vid_path):
    store = pd.HDFStore(vid_path,'r')
    #first, where is the start of the video, in secs?
    start_time = (store.select('df',stop =1)['steve_time'].iloc[0])
    start_seconds = in_seconds(start_time)
    #now for the end of the video. First, check if still same day
    nrows = store.get_storer('df').shape[0]    
    end_time = (store.select('df', start = nrows -1, stop = nrows)['steve_time'].iloc[0])
    end_seconds = in_seconds(end_time)
    store.close()
    return int(start_seconds), int(end_seconds)


"""
Simple conversion of timestamp to total seconds since midnight 
"""
def in_seconds(time):
    return (time.hour * 60 + time.minute) * 60 + time.second



"""
Function to filter bad/faulty data points. The order is important, since (esp. bad feature points) indices are relative to previous filterings
Input: x,y, indices
Output: x,y filtered
"""


def filter_nans(x,y,bad_feat_nans):
    y_no_feat_nans = np.delete(y,bad_feat_nans)
    #then, filter out the nans from label side:
    x_ret = x[:,~np.isnan(y_no_feat_nans)]
    y_ret = y_no_feat_nans[~np.isnan(y_no_feat_nans)]
    return x_ret, y_ret


def filter_artifacts(x,y,artifacts):
    y_no_arts = y[~artifacts]
    x_no_arts = x[:,~artifacts] #keep only the good indices
    return x_no_arts, y_no_arts
    
    
# def filter_data(x,y,feat_ind):
#     #first, filter the NaNs on feature side out on label side as well
#     bad_feat_nan = np.array(feat_ind['NaNs'])
#     bad_feat_nan_inbounds = bad_feat_nan[bad_feat_nan<len(y)] #this shouldn't do anything I think
#     if(len(bad_feat_nan)>len(bad_feat_nan_inbounds)):
#         print('jo diese laengen solltten gleich sein.')
#     y_no_feat_nans = np.delete(y,bad_feat_nan)
#     #now the artifacts
#     bad_feat_artifacts = feat_ind['Artifacts'] 
#     bad_feat_artifacts_inbounds = bad_feat_artifacts[bad_feat_artifacts<len(y_no_feat_nans)]
#     if(len(bad_feat_artifacts)<len(bad_feat_artifacts_inbounds)):
#         print('jo diese laengen sollten auch gleich sein.')
#     y_no_arts = y_no_feat_nans[~bad_feat_artifacts_inbounds]
#     #finally, filter out the nans on label side:
#     x_ret = x[~np.isnan(y_no_arts),:]
#     y_ret = y_no_arts[~np.isnan(y_no_arts)]
#     return x_ret, y_ret

