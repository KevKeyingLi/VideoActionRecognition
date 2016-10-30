#!/usr/bin/env

from pprint import pprint
from scipy import io as sio
import time
import os
import numpy as np
import pickle
import cPickle
import datetime

class Node:
    def __init__(self, start, end, fps, videoname):
        self.id = 0;
        self.start = start
        self.end = end
        self.fps = fps
        self.videoname = videoname 
        self.trajectories = []
        self.features = dict()
        # contains six lists of what features are avaible. This info can be used to compute the histogram
        # use numpy.histogram or scipy.stat.histogram
        self.features['mean_x'] = []
        self.features['mean_y'] = []
        self.features['traj_idx'] = []
        self.features['hog'] = []
        self.features['hof'] = []
        self.features['mbh'] = []
        self.allOverLapLabels = dict() # a dictionary of {overlapping label: [lb_str_t, lb_end_t], overlap}
        self.labels = [] # only the labels that are considered positive.
        # add groud truth
        
    def add_trajectory(self, traj):
        self.trajectories.append(traj)
    def add_label(self, tlabel_info, overlap): # 
        # add to allOverLapLabels
        # tlabel_info = ['video_validation_0000051', [67.5, 75.9], 'Billiards']
        if tlabel_info[2] in self.allOverLapLabels:
            self.allOverLapLabels[tlabel_info[2]].append([tlabel_info[1],overlap])
        else:
            self.allOverLapLabels[tlabel_info[2]] = [[tlabel_info[1],overlap]]
#             self.allOverLapLabels[tlabel_info[2]].append([tlabel_info[1],overlap])
        # add to labels
        if overlap>0.5:
            self.labels.append(tlabel_info[2]) # it is possible that there are multiple same label for a node.
    def record_feature(self):
        ###########
        # this method currently counts all existence, and don't take into account the coverage of each trajectory
        # if need the coverage info in the future, can simply modify to:
        # self.feature_cnt['mean_x'].append([traj.mean_x,traj.coverage])
        for traj in self.trajectories:
            self.features['mean_x'].append(traj.mean_x)
            self.features['mean_y'].append(traj.mean_y)
            self.features['traj_idx'].append(traj.traj_idx)
            self.features['hog'].append(traj.hog)
            self.features['hof'].append(traj.hof)
            self.features['mbh'].append(traj.mbh)
        return
    def set_id(self, idx):
        self.id = idx
class Trajectory:
    def __init__(self, frame_num, mean_x, mean_y, traj_idx, hog, hof, mbh, coverage):
        self.frame_num  = frame_num 
        self.mean_x  = mean_x 
        self.mean_y  = mean_y 
        self.traj_idx  = traj_idx 
        self.hog  = hog
        self.hof  = hof
        self.mbh  = mbh
        self.coverage = coverage # The portion of trajectory included in the window

def computeOverlap(window_start, window_end, label_start, label_end):
    
    if window_start < label_end and label_start < window_end:
        # overlap
        if window_start > label_start:
            l_start = window_start
            s_start = label_start
        else:
            l_start = label_start
            s_start = window_start 
        if window_end > label_end:
            l_end = window_end
            s_end = label_end
        else:
            l_end = label_end
            s_end = window_end 
        return (s_end-l_start)/(l_end - s_start)
    else:
        return 0



def generateNode(video_info, video_tLabelList, windowSize = 150, stepSize = 100):# by frame, default value comes from the Thumos report.
    # generate nodes for a video, return a list of Node
    # before calling this function, require to find the video_info and video_tLabelList for this video. 
    video_name = video_info[0][0]
    duration_frame = video_info[8][0][0]
    fps = float(video_info[9][0])
    # read in the file and form a list of trajectory features
    with open(BASE_DIR+'TH14_validation_features/'+video_name+'.txt','r') as f:
        # later: make the directory as a variable
        trajs = f.readlines()
    trajs = [x.split('\t')[:-1] for x in trajs]
    ####################for debug
#     trajs = trajs[:308001]
    #####################
    traj_start_idx = 0 # the index of the first trajectory of each window
    next_traj_idx = 0
    next_traj_set = False
    # form a list of window start point. 
    frame_step_list = range(1, duration_frame, stepSize)# The start of frame? 
    # Do you also need to consider the window to be full length, not truncated on the last few steps. 
    # Note that in the dataset, there is no trajectory ending on the last frame, so we don't use duration_frame+1
    window_start = 0
    window_end = 0
    node_list = []
    
    for i in range(len(frame_step_list) ):# For each window
        if(frame_step_list[i] + windowSize < duration_frame):
            # Both start and end are inclusive
            window_start = frame_step_list[i]
            window_end = frame_step_list[i] + windowSize - 1
        elif frame_step_list[i] + windowSize < duration_frame+stepSize:
            # The last window
            window_start = frame_step_list[i]
            window_end = duration_frame
        else:
            # Extra windows
            break
#         print window_start,window_end
        
        traj_start_idx = next_traj_idx 
        # next_traj_idx is used to record the start trajectory of next window, when constructing the current node
        next_traj_set = False
        # initialize node
        traj = trajs[traj_start_idx] # Current trajectory
        end_frame = int(traj[0])
        start_frame = end_frame-15+1
        new_node = Node(window_start, window_end, fps, video_name)
        
        # Add trajectories
        while(traj_start_idx<len(trajs) and (window_start<=end_frame<=window_end or window_start<=start_frame<=window_end)):
            coverage = 1.0
            traj = trajs[traj_start_idx] # Current trajectory
            end_frame = int(traj[0])
            start_frame = end_frame-15+1
#             print "start and end"+ str(start_frame)+' , '+str(end_frame)
            if not next_traj_set and i+1 < len(frame_step_list) and end_frame >= frame_step_list[i+1]:
                next_traj_idx = traj_start_idx
                next_traj_set = True
            if(end_frame<= window_end and window_start<=start_frame):
                
                # this trajectory is totally in the window
                
#                 coverage = 1.0
#                 print coverage
                trj_obj = Trajectory(int(traj[0]) , float(traj[1]), float(traj[2]), int(traj[3]), int(traj[4]), int(traj[5]), int(traj[6]), coverage)
                new_node.add_trajectory(trj_obj)
                # add it
            elif start_frame < window_start and window_start <= end_frame: # first few trajs
                # Only the tail of the trajectory is in the window
                coverage = float(end_frame - window_start+1)/15
#                 print coverage
                # add trajectory
                trj_obj = Trajectory(int(traj[0]) , float(traj[1]), float(traj[2]), int(traj[3]), int(traj[4]), int(traj[5]), int(traj[6]), coverage)
                new_node.add_trajectory(trj_obj)
            elif window_end >= start_frame and end_frame>window_end:
                # Only head of the trajectory is in the window
                coverage = float(window_end-(start_frame)+1 )/15
#                 print coverage
                # add trajectory
                trj_obj = Trajectory(int(traj[0]) , float(traj[1]), float(traj[2]), int(traj[3]), int(traj[4]), int(traj[5]), int(traj[6]), coverage)
                new_node.add_trajectory(trj_obj)
            traj_start_idx +=1
        # Generate the feature data, which will be used to generate 4*4000 histogram
        new_node.record_feature()
        # add temporal label to the list:
        window_start_time = window_start * fps
        window_end_time = window_end * fps
        for tlabel_info in video_tLabelList:
            over_lap_score = computeOverlap(window_start_time, window_end_time, tlabel_info[1][0], tlabel_info[1][1])
            if over_lap_score > 0:
                new_node.add_label(tlabel_info, over_lap_score)
        # Add this node into a list. 
        node_list.append(new_node) 
    # return the list of nodes
    for i,node in enumerate(node_list):
        node.set_id(i)
    return node_list


def writeLog(msg):
    msg = str(datetime.datetime.now()) + ':   ' + msg;
    print(msg)
    if not os.path.exists(os.path.dirname(logFileLoc)):
      os.makedirs(os.path.dirname(logFileLoc))
    with open(logFileLoc, 'a') as logFile:
        logFile.write( msg+'\n')

if __name__ == "__main__":
	BASE_DIR = '/Users/baroc/repos/VideoDetection/'
	logFileLoc = BASE_DIR+'generate_graph.log'
	# if this file is imported as a module this part will not be run, since the __name__ will be the module name.
	if not os.path.exists(os.path.dirname(BASE_DIR)):
      print("Please change BASE_DIR in the code.")
      exit()
	# 
	TLBL_DIR = BASE_DIR+'TH14_Temporal_annotations_validation/annotation/' #'./''
	filelist = os.listdir(TLBL_DIR)
	tLabelList = []
	for filename in filelist:
	    if filename.endswith("_val.txt"): 
	        with open(TLBL_DIR+filename,'r') as f:
	            tLabels = f.readlines()
	        tLabels = [x[:-1].split('  ') for x in tLabels]
	        tLabels = [[x[0],map(float, x[1].split(' '))] for x in tLabels]
	        tLabels = [x+[filename[:-8]] for x in tLabels]
	        tLabelList = tLabelList+tLabels
	    else:
	        print('Not a txt file: '+filename)
	tLabelList = sorted(tLabelList)
	videonames = sorted(list(set([x[0] for x in tLabelList])))


	# 
	mat_file_str = BASE_DIR + "validation_set_meta/validation_set_meta/validation_set.mat"
	mat = sio.loadmat(mat_file_str)
	meta_array_1010 = mat['validation_videos'][0] # 1010 entries in meta_array
	id_list = [int(x[-7:])-1 for x in videonames] # a list of zero based indices
	meta_array_200 = meta_array_1010[id_list]
	# the validation data
	for video_info in meta_array_200:
		video_tLabelList = [x for x in tLabelList if x[0]==video_info[0][0]]
		t = time.time()
		node_list = generateNode(video_info, video_tLabelList)#, 100, 50)
		writeLog('node generation of validation data'+video_info[0][0]+ 'took %.2f seconds'% (time.time()-t) )











