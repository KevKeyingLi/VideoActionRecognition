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
	