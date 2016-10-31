#!/usr/bin/env

from pprint import pprint
from scipy import io as sio
import time
import os
import numpy as np
import pickle
import cPickle
import datetime
import node_generator
from node_generator import Node, computeOverlap, generateNode
# class Node:
#     def __init__(self, start, end, fps, videoname):
#         self.id = 0;
#         self.start = start
#         self.end = end
#         self.fps = fps
#         self.videoname = videoname 
#         self.trajectories = []
# #         self.features = dict()
# #         # contains six lists of what features are avaible. This info can be used to compute the histogram
# #         # use numpy.histogram or scipy.stat.histogram
# #         self.features['mean_x'] = []
# #         self.features['mean_y'] = []
# #         self.features['traj_idx'] = []
# #         self.features['hog'] = []
# #         self.features['hof'] = []
# #         self.features['mbh'] = []
#         self.histogram = []
#         self.allOverLapLabels = dict() # a dictionary of {overlapping label: [lb_str_t, lb_end_t], overlap}
#         self.labels = [] # only the labels that are considered positive.
#         # add groud truth
        
#     def add_trajectory(self, traj): # call this when you need it
#         self.trajectories.append(traj)
#     def add_label(self, tlabel_info, overlap): # 
#         # add to allOverLapLabels
#         # tlabel_info = ['video_validation_0000051', [67.5, 75.9], 'Billiards']
#         if tlabel_info[2] in self.allOverLapLabels:
#             self.allOverLapLabels[tlabel_info[2]].append([tlabel_info[1],overlap])
#         else:
#             self.allOverLapLabels[tlabel_info[2]] = [[tlabel_info[1],overlap]]
# #             self.allOverLapLabels[tlabel_info[2]].append([tlabel_info[1],overlap])
#         # add to labels
#         if overlap>0.5:
#             self.labels.append(tlabel_info[2]) # it is possible that there are multiple same label for a node.
#     def record_feature(self): 
#         ###########
#         # this method currently counts all existence, and don't take into account the coverage of each trajectory
#         # if need the coverage info in the future, can simply modify to:
#         # self.feature_cnt['mean_x'].append([traj.mean_x,traj.coverage])
        
#         if len(self.features['traj_idx']) == 0:
#             raise ValueError("record_features not usable. Because no features. Use add_feature in loop instead. ")
#         for traj in self.trajectories:
#             self.features['mean_x'].append(traj.mean_x)
#             self.features['mean_y'].append(traj.mean_y)
#             self.features['traj_idx'].append(traj.traj_idx)
#             self.features['hog'].append(traj.hog)
#             self.features['hof'].append(traj.hof)
#             self.features['mbh'].append(traj.mbh)
#         return
    
#     def add_feature(self, mean_x, mean_y, traj, hog, hof, mbh):
#         # this function computes and adds the histogram of the 16,000 features, 
#         # and adds the mean_x mean_y information if necessary
#         hist_hog = np.histogram(hog,4000,(0,4000))[0]
#         hist_hof = np.histogram(hof,4000,(0,4000))[0]
#         hist_mbh = np.histogram(mbh,4000,(0,4000))[0]
#         hist_traj = np.histogram(traj,4000,(0,4000))[0]
#         self.histogram = np.concatenate((hist_traj, hist_hog, hist_hof, hist_mbh))
# #         self.mean_x = mean_x
# #         self.mean_y = mean_y
#     def set_id(self, idx):
#         self.id = idx

# class Trajectory:
#     def __init__(self, frame_num, mean_x, mean_y, traj_idx, hog, hof, mbh, coverage):
#         self.frame_num  = frame_num 
#         self.mean_x  = mean_x 
#         self.mean_y  = mean_y 
#         self.traj_idx  = traj_idx 
#         self.hog  = hog
#         self.hof  = hof
#         self.mbh  = mbh
#         self.coverage = coverage # The portion of trajectory included in the window


# def computeOverlap(window_start, window_end, label_start, label_end):
    
#     if window_start < label_end and label_start < window_end:
#         # overlap
#         if window_start > label_start:
#             l_start = window_start
#             s_start = label_start
#         else:
#             l_start = label_start
#             s_start = window_start 
#         if window_end > label_end:
#             l_end = window_end
#             s_end = label_end
#         else:
#             l_end = label_end
#             s_end = window_end 
#         return (s_end-l_start)/(l_end - s_start)
#     else:
#         return 0


def writeLog(msg):
    msg = str(datetime.datetime.now()) + ':   ' + msg;
    print(msg)
    
    if not os.path.exists(os.path.dirname(logFileLoc)):
      os.makedirs(os.path.dirname(logFileLoc))
    with open(logFileLoc, 'a') as logFile:
        logFile.write( msg+'\n')

if __name__ == "__main__":
	startT = time.time()
	BASE_DIR = '/Users/baroc/repos/VideoActionRecognition/'
	logFileLoc = BASE_DIR+'generate_graph.log'
	# if this file is imported as a module this part will not be run, since the __name__ will be the module name.
	if not os.path.exists(os.path.dirname(BASE_DIR)):
		print("Please change BASE_DIR in the code.")
		exit()
	
	# load the temporal list
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
	node_list = []
	mat_file_str = BASE_DIR + "validation_set_meta/validation_set_meta/validation_set.mat"
	mat = sio.loadmat(mat_file_str)
	meta_array_1010 = mat['validation_videos'][0] # 1010 entries in meta_array
	id_list = [int(x[-7:])-1 for x in videonames] # a list of zero based indices
	meta_array_200 = meta_array_1010[id_list]
	# the validation data
	i = 1
	for video_info in meta_array_200:
		print("Start generate node for the "+str(i)+"th video of Validation data" )
		video_tLabelList = [x for x in tLabelList if x[0]==video_info[0][0]]
		t = time.time()
		temp_node_list = generateNode(video_info, video_tLabelList, BASE_DIR)#,0.5 , 100, 50)
		writeLog('Node generation of validation data '+video_info[0][0]+ ' took %.2f seconds'% (time.time()-t) )
		node_list = node_list + temp_node_list
		i += 1
	
	t = time.time()
	cPickle.dump( node_list, open( BASE_DIR+"validation_video_nodes.p", "wb" ), protocol=cPickle.HIGHEST_PROTOCOL )
	print(time.time()-t) 

	endT = time.time()
	writeLog("*** \n\nNode generation finished, generated %d nodes in total, time elapsed %.2f seconds" % (len(node_list),endT-startT) )
	# cPFile = open(BASE_DIR+"validation_video_nodes.p", 'rb')
	# t = time.time()
	# cP = cPickle.load(cPFile)
	# print(time.time()-t)
	# cPFile.close()










