import os
import numpy as np
from scipy import io as sio
import time
from node_generator import Node, export_mat
import cPickle
import datetime

def writeLog(msg):
    msg = str(datetime.datetime.now()) + ':   ' + msg;
    print(msg)
    
    if not os.path.exists(os.path.dirname(logFileLoc)):
      os.makedirs(os.path.dirname(logFileLoc))
    with open(logFileLoc, 'a') as logFile:
        logFile.write( msg+'\n')


startT = time.time()
BASE_DIR = '/data/UCF/data/Thumos/iDTF/'#'/Users/baroc/repos/VideoActionRecognition/'
OUTPUT_DIR = BASE_DIR+'Keying/'
logFileLoc = BASE_DIR+'exporting_validation_node_mats.log'
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
t = time.time()
export_mat(node_list, ['label','feature'], [OUTPUT_DIR+'validation_node_labels',OUTPUT_DIR+'validation_node_features'],[1,1])
writeLog("Finish both matrices in %.2f secondes. "% (time.time() - t))