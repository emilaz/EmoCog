import numpy as np
import pandas as pd
import os

"""
Given a day, finds the corresponding video file I guess
"""


def find_paths(patient_name, day_no):
    # create filename:
    ####
    filename = patient_name + '_fullday_' + str(day_no) + '.h5'
    video_filename = patient_name + '_day_' + str(day_no) + '.hdf'
    path = os.path.join('/nas/ecog_project/derived/processed_ecog', patient_name, 'full_day_ecog')
    ecog_path = os.path.join(path, filename)
    video_path = os.path.join('/home/emil/data/hdf_data/by_day_new/', patient_name, video_filename)
    ###
    # filename = patient_name+'_ecog_day_'+str(day_no)+'.h5'
    # video_filename = patient_name+'_day_'+str(day_no)+'.hdf'
    # ecog_path = os.path.join('/home/emil/data/hdf_data/by_day_new/fake',patient_name,filename)
    # video_path = os.path.join('/home/emil/data/hdf_data/by_day_new/fake',patient_name,video_filename)
    ###
    # check if file exists
    if not (os.path.exists(ecog_path) and os.path.exists(video_path)):
        FileNotFoundError('Files for day {} do not exist. Check if day or patient name is wrong.'.format(day_no))
        return
    return ecog_path, video_path


"""
This function finds out how many seconds into the day the first session of day started and when it ended
Input: Path to video session hdf file
Output: Seconds passed since midnight until video session began
"""


def find_start_and_end_time(vid_path):
    store = pd.HDFStore(vid_path, 'r')
    # first, where is the start of the video, in secs?
    start_time = (store.select('df', stop=1)['steve_time'].iloc[0])
    start_seconds = in_seconds(start_time)
    # now for the end of the video. First, check if still same day
    nrows = store.get_storer('df').shape[0]
    # this is beginning of last video. since we might be going into next day, we won't use the last vid
    end_time = (store.select('df', start=nrows - 1, stop=nrows)['steve_time'].iloc[0])
    remaining_frames = (store.select('df', start=nrows - 1, stop=nrows)['frame'].iloc[0])
    end_seconds = in_seconds(end_time) + remaining_frames / 30
    store.close()
    return int(start_seconds), int(end_seconds)


"""
Simple conversion of timestamp to total seconds since midnight 
"""


def in_seconds(time):
    return (time.hour * 60 + time.minute) * 60 + time.second + time.microsecond * 1e-6


"""
Function to filter bad/faulty data points. 
Input: df containing x,y and bad indices
Output: x,y filtered. concatenated over days and patients (day-first ordering)
"""


def filter_nans(df):
    # filter the bad feature indices for each day on label side
    #### old
    # y_no_feat_nans = np.delete(y, bad_feat_nans) # deprecated
    # then, filter out the nans from label side:
    # x_ret = x[:, ~np.isnan(y_no_feat_nans)]
    # y_ret = y_no_feat_nans[~np.isnan(y_no_feat_nans)]
    ####
    # first, filter the bad feature indices on label side for each patient,day
    df['Y_no_bad'] = [np.array([j for ix, j in enumerate(i[4]) if ix not in i[3]]) for i in df.values]
    x_clean = []
    y_clean = []
    # now, delete nans from labels on both side
    for idx, row in df.iterrows():
        non_nans = ~np.isnan(row['Y_no_bad'])
        x_row = row['X'][:, non_nans]
        y_row = row['Y_no_bad'][non_nans]
        x_clean.append(x_row)
        y_clean.append(y_row)
    x_ret = np.concatenate(x_clean,axis=1)
    y_ret = np.concatenate(y_clean)

    return x_ret, y_ret


def filter_artifacts(x, y, artifacts):
    y_no_arts = y[~artifacts]
    x_no_arts = x[:, ~artifacts]  # keep only the good indices
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
