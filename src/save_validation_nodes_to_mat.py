import os
import numpy as np
from scipy import io as sio
import time
from node_generator import Node, export_mat

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

export_mat(node_list, ['label','feature'], [OUTPUT_DIR+'validation_node_labels.mat',OUTPUT_DIR+'validation_node_features.mat']])