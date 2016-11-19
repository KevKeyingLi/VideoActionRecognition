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
	BASE_DIR = '/data/UCF/data/Thumos/iDTF/'#'/Users/baroc/repos/VideoActionRecognition/'
	OUTPUT_DIR = BASE_DIR+'Keying/'
	logFileLoc = BASE_DIR+'generate_test_node.log'
	# if this file is imported as a module this part will not be run, since the __name__ will be the module name.
	if not os.path.exists(os.path.dirname(BASE_DIR)):
		print("Please change BASE_DIR in the code.")
		exit()
	
	# load the temporal list
	TEST_TLBL_DIR = BASE_DIR+'TH14_Temporal_Annotations_Test/annotations/annotation/' #'./''
	test_filelist = os.listdir(TEST_TLBL_DIR)
	test_tLabelList = []
	for filename in test_filelist:
	    if filename.endswith("_test.txt"): 
	        with open(TEST_TLBL_DIR+filename,'r') as f:
	            tLabels = f.readlines()
	        tLabels = [x[:-1].split('  ') for x in tLabels]
	        tLabels = [[x[0],map(float, x[1].split(' '))] for x in tLabels]
	        tLabels = [x+[filename[:-9]] for x in tLabels]
	        test_tLabelList = test_tLabelList+tLabels
	    else:
	        print('Not a txt file: '+filename)
	test_tLabelList = sorted(test_tLabelList)
	test_videonames = sorted(list(set([x[0] for x in test_tLabelList])))



	# 
	test_node_list = []
	test_mat_file_str=BASE_DIR+"test_set_meta.mat"
	mat = sio.loadmat(test_mat_file_str)
	test_meta_data = mat['test_videos'][0]
	# the test data
	i = 1
	for video_info in test_meta_data: 
		print("Start generate node for the "+str(i)+"th video of Test data" )
		video_test_tLabelList = [x for x in test_tLabelList if x[0]==video_info[0][0]]
		t = time.time()
		temp_test_node_list = generateNode(video_info, video_test_tLabelList, BASE_DIR + 'TH14_test_features/')#,0.5 , 100, 50)
		writeLog('Node generation of test data '+video_info[0][0]+ ' took %.2f seconds'% (time.time()-t) )
		test_node_list = test_node_list + temp_test_node_list
		i += 1
	for i,node in enumerate(node_list):
		node.set_id(i)
	t = time.time()
	cPickle.dump( test_node_list, open( OUTPUT_DIR+"test_video_nodes.p", "wb" ), protocol=cPickle.HIGHEST_PROTOCOL )
	print(time.time()-t) 

	endT = time.time()
	writeLog("*** \n\nNode generation finished, generated %d nodes in total, time elapsed %.2f seconds" % (len(test_node_list),endT-startT) )



