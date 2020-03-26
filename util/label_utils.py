import numpy as np
from util.sync_utils import in_seconds


"""
Converts strings into int/np.nan. Necessary for if looking at annotations in dataframe
Input: Data
Output: Mal gukcen
"""

def convert_labels_readable(annot):
    nan_indices = annot == 'N/A'
    annot[annot != 'Happy'] = 0
    annot[annot == 'Happy'] = 1
    annot[nan_indices] = np.nan
    return annot


"""
Converts labels in range [0,1] into binary Happy/Not Happy, given cutoff
Input: Labels, cutoff
Output: Binary Labels
"""


def do_cutoff(y_in, cutoff):
    y = y_in.copy()
    y[y > cutoff] = 1
    y[y < 1] = 0
    return y


"""
For a given video, finds the number of frames the video is supposed to have.
Input: The hdf dataframe from Gautham's network, the given video
Output: Number of frames the video should have
"""

# def find_number_frames(df, vid, last_vid, fill_vid_after):
#     next_vid = str(int(vid)+1) #which is the next number?
#     len_vid = len(next_vid)
#     next_vid = '0'*(4-len_vid)+next_vid
#     time_vid = df[df['vid']==vid]['datetime'].iloc[0]
#     try:
#         time_next_vid = df[df['vid']==next_vid]['datetime'].iloc[0]
#         time = datetime.datetime.strptime(time_vid, '%Y-%m-%d %H:%M:%S.%f')
#         new_time = datetime.datetime.strptime(time_next_vid, '%Y-%m-%d %H:%M:%S.%f')
#         elapsed = (new_time-time).total_seconds()
#     except: #in case of last video
#         if last_vid != vid:
#             print('this is not the last video, but after, theres a missing one:',vid)
#             fill_vid_after[0] = True
#         elapsed = 120
#     elapsed_frames = int(elapsed *30)
#     return elapsed_frames

"""
For a given video, finds the number of frames passed b/w this and the next video.
Input: The hdf dataframe from Gautham's network, the given video
Output: Number of frames between
"""


def find_number_frames(df, vid, last_vid):
    time_vid = df[df['video'] == vid]['steve_time'].iloc[0]
    time_vid_secs = in_seconds(time_vid)
    next_vid = vid
    while True:
        next_vid = str(int(next_vid)+1)  #this is to construct videoname of next vid
        len_vid = len(next_vid)
        next_vid = '0'*(4-len_vid)+next_vid
        try:
            time_next_vid = df[df['video']==next_vid]['steve_time'].iloc[0]
            time_next_vid_secs = in_seconds(time_next_vid)
            elapsed = (time_next_vid_secs-time_vid_secs)
            break
        except IndexError: #in case of last video
            if last_vid == vid:
                elapsed = float(df[df['video']==vid]['frame'].iloc[-1])/30
                break
    elapsed_frames = elapsed * 30
    time_vid_frames = time_vid_secs * 30
    return int(time_vid_frames), int(elapsed_frames)

"""
This function distributes the given m frames along an array of size n (>m), the rest are nans
Input: The given labels for some video, the number of frames(and therefore labels) the video should have
Output: Array of length n with frames distributed along array (in ordered manner!)"""

def fill_frames(actual_labels,supposed_no_labels):
    no_labels = len(actual_labels) #how many labels do we have?
    # max_of_both = max(no_labels,supposed_no_labels) #warning: this only temporarily hopefully
    places = np.empty(supposed_no_labels) #create array of length we actually want
    # places = np.empty(max_of_both) #create array of length we actually want
    places[:]=np.nan
    #all vids are around 120 long, if not, then there are vids missing/pause bw sessions
    if supposed_no_labels > 200*30:
        #fill only 2 mins
        positions = sorted(np.random.choice(max(no_labels, 121*30),no_labels,replace=False))
    elif no_labels<supposed_no_labels:
        positions = sorted(np.random.choice(supposed_no_labels,no_labels,replace=False)) #which positions do we want to fill?
    else:
        positions = np.arange(supposed_no_labels)
        actual_labels = actual_labels[:supposed_no_labels]
        if no_labels-supposed_no_labels > 10:
            print('ajoo', no_labels-supposed_no_labels)
            raise ValueError('More than 10 frames are overshot here, somethings wrong')
    try:
        places[positions] = actual_labels
    except IndexError as e:
        print(e)
        print(supposed_no_labels)
        print(actual_labels)
        print(positions)

    return places 
