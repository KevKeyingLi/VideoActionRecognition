import os
import numpy as np
from scipy import io as sio

class Node:
    def __init__(self, start, end, fps, videoname):
        self.id = 0;
        self.start = start
        self.end = end
        self.fps = fps
        self.videoname = videoname 
        self.trajectories = []
#         self.features = dict()
#         # contains six lists of what features are avaible. This info can be used to compute the histogram
#         # use numpy.histogram or scipy.stat.histogram
#         self.features['mean_x'] = []
#         self.features['mean_y'] = []
#         self.features['traj_idx'] = []
#         self.features['hog'] = []
#         self.features['hof'] = []
#         self.features['mbh'] = []
        self.histogram = []
        self.allOverLapLabels = dict() # a dictionary of {overlapping label: [lb_str_t, lb_end_t], overlap}
        self.labels = [] # only the labels that are considered positive.
        # add groud truth
        
    def add_trajectory(self, traj): # call this when you need it
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
        # this method is not used currently
        # counts all existence, and don't take into account the coverage of each trajectory
        # if need the coverage info in the future, can simply modify to:
        # self.feature_cnt['mean_x'].append([traj.mean_x,traj.coverage])
        
        if len(self.features['traj_idx']) == 0:
            raise ValueError("record_features not usable. Because no features. Use add_feature in loop instead. ")
        for traj in self.trajectories:
            self.features['mean_x'].append(traj.mean_x)
            self.features['mean_y'].append(traj.mean_y)
            self.features['traj_idx'].append(traj.traj_idx)
            self.features['hog'].append(traj.hog)
            self.features['hof'].append(traj.hof)
            self.features['mbh'].append(traj.mbh)
        return
    
    def add_feature(self, mean_x, mean_y, traj, hog, hof, mbh):
        # this function computes and adds the histogram of the 16,000 features, 
        # and adds the mean_x mean_y information if necessary
        hist_hog = np.histogram(hog,4000,(0,4000))[0]
        hist_hof = np.histogram(hof,4000,(0,4000))[0]
        hist_mbh = np.histogram(mbh,4000,(0,4000))[0]
        hist_traj = np.histogram(traj,4000,(0,4000))[0]
        self.histogram = np.concatenate((hist_traj, hist_hog, hist_hof, hist_mbh))
#         self.mean_x = mean_x
#         self.mean_y = mean_y
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
    IOU
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

def generateNode(video_info, video_tLabelList, FEATURE_DIR, traj_coverage_threashold = 0.5,windowSize = 150, stepSize = 100):# by frame, default value comes from the Thumos report.
    # generate nodes for a video, return a list of Node
    # before calling this function, require to find the video_info and video_tLabelList for this video. 
    video_name = video_info[0][0]
    if 'validation' in video_name:
        duration_frame = video_info[8][0][0]
        fps = float(video_info[9][0][0])
    elif 'test' in video_name:
        duration_frame = int(video_info[5][0][0]*video_info[7][0][0])
        fps = float(video_info[7][0][0])
    # read in the file and form a list of trajectory features
    with open(FEATURE_DIR+video_name+'.txt','r') as f:
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
    frame_step_list = range(0, duration_frame, stepSize)# The start of frame? 
    # Do you also need to consider the window to be full length, not truncated on the last few steps. 
    # Note that in the dataset, there is no trajectory ending on the last frame, so we don't use duration_frame+1
    window_start_frame = 0
    window_end_frame = 0
    node_list = []
    
    for i in range(len(frame_step_list) ):# For each window
        if(frame_step_list[i] + windowSize < duration_frame):
            # Both start and end are inclusive
            window_start_frame = frame_step_list[i]
            window_end_frame = frame_step_list[i] + windowSize - 1
        elif frame_step_list[i] + windowSize < duration_frame+stepSize:
            # The last window
            window_start_frame = frame_step_list[i]
            window_end_frame = duration_frame
        else:
            # Extra windows
            break
#         print window_start_frame,window_end_frame
        
        traj_start_idx = next_traj_idx 
        # next_traj_idx is used to record the start trajectory of next window, when constructing the current node
        next_traj_set = False
        # initialize node
        traj = trajs[traj_start_idx] # Current trajectory
        end_frame = int(traj[0])
        start_frame = end_frame-15+1
        new_node = Node(window_start_frame, window_end_frame, fps, video_name)
        traj_len = 15
        # feature list
        features = dict()
        features['mean_x'] = []
        features['mean_y'] = []
        features['traj'] = []
        features['hog'] = []
        features['hof'] = []
        features['mbh'] = []
        # Add trajectories, 
        while(traj_start_idx<len(trajs) and (window_start_frame<=end_frame<=window_end_frame or window_start_frame<=start_frame<=window_end_frame)):
            coverage = 1.0
            traj = trajs[traj_start_idx] # Current trajectory
            end_frame = int(traj[0])
            start_frame = end_frame-15+1
#             print "start and end"+ str(start_frame)+' , '+str(end_frame)
            if not next_traj_set and i+1 < len(frame_step_list) and end_frame >= frame_step_list[i+1]:
                next_traj_idx = traj_start_idx
                next_traj_set = True
            if(end_frame<= window_end_frame and window_start_frame<=start_frame):
                pass
            # this trajectory is totally in the window
            #     coverage = 1.0
            elif start_frame < window_start_frame and window_start_frame <= end_frame: # first few trajs
                # Only the tail of the trajectory is in the window
                coverage = float(end_frame - window_start_frame + 1)/traj_len
#                 print coverage
            elif window_end_frame >= start_frame and end_frame>window_end_frame:
                # Only head of the trajectory is in the window
                coverage = float(window_end_frame-(start_frame)+1 )/traj_len
#                 print coverage
            # add trajectory #         add_trajectory call it only when you need it!!!
#             trj_obj = Trajectory(int(traj[0]) , float(traj[1]), float(traj[2]), int(traj[3]), int(traj[4]), int(traj[5]), int(traj[6]), coverage)
#             new_node.add_trajectory(trj_obj)
                    
            # Generate the feature data, which will be used to generate 4*4000 histogram
            if coverage > traj_coverage_threashold:
                features['mean_x'].append(float(traj[1]))
                features['mean_y'].append(float(traj[2]))
                features['traj'].append(int(traj[3]))
                features['hog'].append(int(traj[4]))
                features['hof'].append(int(traj[5]))
                features['mbh'].append(int(traj[6]))
            traj_start_idx +=1
#         new_node.record_feature() # obsolete function

        # add temporal label to the list:
        window_start_time = window_start_frame/fps
        window_end_time = window_end_frame/fps
        for tlabel_info in video_tLabelList: 
            over_lap_score = computeOverlap(window_start_time, window_end_time, tlabel_info[1][0], tlabel_info[1][1])
            if over_lap_score > 0:
                new_node.add_label(tlabel_info, over_lap_score)
        # compute and add histogram
        new_node.add_feature(features['mean_x'],features['mean_y'],features['traj'],features['hog'],features['hof'],features['mbh'])
        # Add this node into a list. 
        node_list.append(new_node) 
    # return the list of nodes
    return node_list

{
        '': 1,
        'b': 2,
}[x]

label_index_UCF = {
'BaseballPitch':7,
'BasketballDunk':9,
'Billiards':12,
'CleanAndJerk':21,
'CliffDiving':22,
'CricketBowling':23,
'CricketShot':24,
'Diving':26,
'FrisbeeCatch':31,
'GolfSwing':33,
'HammerThrow':36,
'HighJump':40,
'JavelinThrow':45,
'LongJump':51,
'PoleVault':68,
'Shotput':79,
'SoccerPenalty':85,
'TennisSwing':92,
'ThrowDiscus':93,
'VolleyballSpiking':97
}

label_index_21 = {
'BaseballPitch':0,'BasketballDunk':1,'Billiards':2,'CleanAndJerk':3,'CliffDiving':4,'CricketBowling':5,'CricketShot':6,'Diving':7,'FrisbeeCatch':8,'GolfSwing':9,'HammerThrow':10,'HighJump':11,'JavelinThrow':12,'LongJump':13,'PoleVault':14,'Shotput':15,'SoccerPenalty':16,'TennisSwing':17,'ThrowDiscus':18,'VolleyballSpiking':19,'Ambiguous':20
}

def export_mat(node_list, item_list, mat_files):
# item_list is a list of what to output
#   namely: 'feature', 'label'
# mat_files is a list of directories to out put,
#   It is the same size of the item_list
    item_num = len(item_list)
    if item_num != len(mat_files):
        print("Bad parameters for export_mat, size of item_list and mat_files does not match. ")
        return 
    node_num = len(node_list)
    arrays = dict()
    for item in item_list:
        if item == 'label':
            arrays[item] =  np.zeros([len(node_list),21])
        elif: item == 'feature':
            arrays[item] =  np.zeros([len(node_list),16000])
        else:
            item_list.remove(item)
            print("No such item, removed!")
    if 'label' in item_list:
        #
        i = 0
        for node in node_list:
            for label in node.labels:
                OL_Label_List = node.allOverLapLabels[label]
                # print(max(OL_Label_List, key = lambda x : x[1]))
                arrays['label'][i][label_index_21[label]] = max(OL_Label_List, key = lambda x : x[1])[1]
            i += 1
    if 'feature' in item_list:
        i = 0
        for node in node_list:
            arrays['feature'][i] = node.histogram
            i += 1
    for i in range(len(item_list)):
        sio.savemat(mat_files[i], {item_list[i]+'_matrix':arrays[item_list[i]]});
    return




