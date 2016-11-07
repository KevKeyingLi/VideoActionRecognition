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
		temp_node_list = generateNode(video_info, video_tLabelList, BASE_DIR + 'TH14_validation_features/')#,0.5 , 100, 50)
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










