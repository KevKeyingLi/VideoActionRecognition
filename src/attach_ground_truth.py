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
labels = attach_ground_truth(BASE_DIR, window_list, True):  # High level interface containing all the detail funcitonalities in one

# videonames = sorted(list(set([x[0] for x in tLabelList])))
# tLabelList:
# [['video_validation_0000051', [67.5, 75.9], 'Billiards'],
#  ['video_validation_0000051', [85.9, 90.6], 'Billiards'],
#  ['video_validation_0000051', [139.3, 148.2], 'Billiards'],
#  ['video_validation_0000052', [24.3, 24.8], 'Billiards'],
#  ['video_validation_0000053', [9.1, 13.8], 'Billiards'],
#  ...
#  ['video_validation_0000162', [152.5, 155.1], 'Diving'],
#  ['video_validation_0000162', [155.8, 158.5], 'CliffDiving'],
#  ['video_validation_0000162', [155.8, 158.5], 'Diving'],
#  ['video_validation_0000162', [163.0, 164.0], 'Ambiguous'],
#  ['video_validation_0000162', [164.1, 167.1], 'CliffDiving']...]
