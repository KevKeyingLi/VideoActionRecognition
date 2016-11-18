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
import math
import heapq as pq
from operator import itemgetter

T_THRD = 0 #float(1)/3

def temporal_distance(nodeA, nodeB, other):
    # Larger means more overlap: 0-1, simply percentage of the window
    if nodeA.videoname != nodeB.videoname:
    	return 0
    # if nodeA.start < nodeB.end and nodeB.start < nodeA.end:
        # overlap
    if nodeA.start < nodeB.start:
        lstart = nodeB.start
        sstart = nodeA.start
    else:
        lstart = nodeA.start
        sstart = nodeB.start
    if nodeA.end < nodeB.end:
        lend = nodeB.end
        send = nodeA.end
    else:
        lend = nodeA.end
        send = nodeB.end
    # IOU
    IOU = float(send - lstart)/(lend - sstart)
    return IOU if IOU >0 else 0
    # return float(send - lstart)/(nodeA.end - nodeA.start)
    # return float(nodeA.end - nodeA.start)/(lend - sstart)
    # else:
    #     return 0

def feature_distance(nodeA, nodeB, sigma = 2000):
#     l2_norm_2 = reduce(lambda x, y: x+y, [x**2 for x in nodeA.histogram - nodeB.histogram])
    l2_norm = np.linalg.norm(nodeA.histogram - nodeB.histogram)
    return math.exp(-(l2_norm/(sigma))**2)

def build_edge(node_list, distance_function, sigma = 2000, k=10): # top K
    i = 0
    adj_list = []
    while i < len(node_list):
        j = i+1
        print("Starting Node: %d" % i)
        t = time.time()
        while j<len(node_list):
#             print i, j
            if i == 0:
                adj_list.append([])
                
                if j == 1:
                    adj_list.append([])
#                 print adj_list
            d = distance_function(node_list[i],node_list[j], sigma)
            if distance_function == feature_distance:
                if len(adj_list[i])<k: #top K
                    pq.heappush(adj_list[i],(d,j) )
                else:
                    pq.heappushpop(adj_list[i],(d,j))
                if len(adj_list[j])<k: #top K
                    pq.heappush(adj_list[j],(d,i) )
                else:
                    pq.heappushpop(adj_list[j],(d,i))
#                 print("feature distance chosen")

            elif distance_function == temporal_distance:
#                 print("temporal distance chosen")
                
                if d > T_THRD: # The criteria to add edge
                    adj_list[i].append((d,j))
                    adj_list[j].append((d,i))
            
            else:
                print("No such distance function")
                return
            j += 1
        writeLog("Finished node %d after %.2f seconds."%(i,time.time()-t))
        i += 1
    if distance_function == feature_distance:
    	# make symmetric
        make_symmetric(adj_list)
        # print "sorting"
        i = 0
        for edges in adj_list:
            adj_list[i] = sorted(edges, key=itemgetter(1))
            i +=1

    return adj_list

def make_symmetric(adj_list):
	t = time.time()
	i = 0
	for node in adj_list:
		for edge in node:
			j = edge[1]
			if i not in [ e[1] for e in adj_list[j] ]:
				adj_list[j].append((edge[0],i))
		i += 1
	print("make_symmetric taking %.2f secs" % (time.time()-t))
	return


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
    logFileLoc = BASE_DIR+'generate_edges.log'
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

