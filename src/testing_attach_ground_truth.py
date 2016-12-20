import os
import numpy as np
from node_generator import attach_ground_truth

def load_window_list():
	return generate_test_window_list()


def generate_test_window_list():
	l = list()
	for i in range(300):
		l.append(('video_validation_0000051',i*16,(i+1)*16-1))
	return l


BASE_DIR = '/Users/baroc/repos/VideoActionRecognition/'
window_list = load_window_list()# window list is a list of tupels(videopathname, start frame, end frame)
labels = attach_ground_truth(BASE_DIR, window_list, True)  # High level interface containing all the detail funcitonalities in one
