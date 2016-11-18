#!/usr/bin/env

# from pprint import pprint
# from scipy import io as sio
import time
import os
import numpy as np
# import pickle
import cPickle
import node_generator
from node_generator import Node, computeOverlap, generateNode
from edge_generator import temporal_distance,feature_distance,build_edge,writeLog




if __name__ == "__main__":
    startT = time.time()
    BASE_DIR = '/data/UCF/data/Thumos/iDTF/'#'/Users/baroc/repos/VideoActionRecognition/'
    OUTPUT_DIR = BASE_DIR+'Keying/'
    # logFileLoc = BASE_DIR+'generate_edges.log'
    # if this file is imported as a module this part will not be run, since the __name__ will be the module name.
    if not os.path.exists(os.path.dirname(OUTPUT_DIR)):
    	print("Please change OUTPUT_DIR in the code.")
    	exit()
    # Load pickle
    cPFile = open(OUTPUT_DIR+"validation_video_nodes.p", 'rb')
    t = time.time()
    node_list = cPickle.load(cPFile)
    # time.time()-t
    writeLog("Load pickle file in %.2f secondes. "% (time.time() - t))


# Generate edges using similarity
    feature_startT = time.time()
    t = time.time()
    # adj_list =[]
    adj_list = build_edge(node_list, feature_distance, sigma = 2000, k= 7)
    writeLog("Generate edges took %.2f secondes. "% (time.time() - t))
    t = time.time()
    cPickle.dump( adj_list, open( OUTPUT_DIR + "val_adj_list_by_feature_k_7_sigma_2000.p", "wb" ), protocol=cPickle.HIGHEST_PROTOCOL )
    writeLog("write to file took %.2f seconds"% (time.time()-t)) # 14.753661871 0.0874960422516
    t = time.time() - feature_startT
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    writeLog("Feature edges finished after %02d:%02d:%02d" % (h, m, s)) # 14.753661871 0.0874960422516
	
# 2016-10-31 11:37:45.191174:   Generate edges took 3330.18 secondes. 
# 2016-10-31 11:37:45.373308:   write to file took 0.18 seconds
# 2016-10-31 11:37:45.374234:   Feature edges finished after 00:55:30
# generate edge usign temporal info

    feature_startT = time.time()
    t = time.time()
    # adj_list =[]
    adj_list = build_edge(node_list, temporal_distance)
    writeLog("Generate edges took %.2f secondes. "% (time.time() - t))
    t = time.time()
    cPickle.dump( adj_list, open( OUTPUT_DIR + "val_adj_list_by_temporal.p", "wb" ), protocol=cPickle.HIGHEST_PROTOCOL )
    writeLog("write to file took %.2f seconds"% (time.time()-t)) # 14.753661871 0.0874960422516
    t = time.time()-feature_startT
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    writeLog("Temporal edges finished after %02d:%02d:%02d" % (h, m, s))
    t = time.time() - startT
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    writeLog("Program finished after %02d:%02d:%02d" % (h, m, s))

