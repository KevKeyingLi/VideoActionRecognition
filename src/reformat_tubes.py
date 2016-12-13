from scipy import io as sio
import numpy as np
from pprint import pprint
import os
import cPickle
import time 


BASE_DIR = '../'
OBJ_DIR = BASE_DIR+'ss_optiflow_all/'
OUTPUT_DIR = BASE_DIR+'output/'



scales = []
def reformat(files_dict):
    scales = [dict(),dict()] # same size as the number of scales
    l = 1
    for videoname in files_dict:
    	# print('Start '+str(l))
    	t = time.time()
        if not files_dict[videoname]:
            for scale in scales:
                scale[videoname] = None
        else:
            mat_file = files_dict[videoname][0]
            try:
                mat = sio.loadmat(mat_file)
            except Exception as e:
            	for scale in scales:
                    scale[videoname] = None
                print('Error caught on file: '+videoname)
                print(files_dict[videoname])
                print(e)
                continue
            cltr = mat['cltr']
            for i in range(len(cltr[0])):
                scales[i][videoname]=[]
                for j in range(len(cltr[0,i][0])):
                    scales[i][videoname].append(dict())
                    scales[i][videoname][j]["start"] = cltr[0,i][0,j]['start'][0,0]
                    scales[i][videoname][j]["end"] = cltr[0,i][0,j]['end'][0,0]
                    scales[i][videoname][j]["length"] = cltr[0,i][0,j]['length'][0,0]
                    scales[i][videoname][j]["avgsel"] = cltr[0,i][0,j]['clusters']['avgsel'][0] # a length * 4 array
    	# print('Finished '+str(l)+' after %.2f s' %(time.time()-t))
    	l +=1
    return scales

dir_list = os.listdir(OBJ_DIR)
dir_list = [OBJ_DIR+dir+'/' for dir in dir_list]
file_names = []
files = dict()
eptfiles = []
multfiles = dict()
for dir in dir_list:
    if not os.path.isdir(dir):
        continue
    file_list = os.listdir(dir)
    for filename in file_list:
        if not os.path.isdir(dir+filename+'/'):
            print('Not dir')
            print(dir+filename+'/')
            print('----')
            continue
        mat_list = os.listdir(dir+filename+'/')
        if len(mat_list) == 0:
            files[filename] = None
            eptfiles.append(filename)
        else:
            file_names.append(filename)
            files[filename] = [dir+filename+'/'+ mat for mat in mat_list if '.mat' in mat and not mat.startswith('.') ]
            if len(mat_list) >1:
                print('more than one file:')
                print(filename)
                multfiles[filename] = len(mat_list)
print('Start reformatting')
scales = reformat(files)
print('Finished reformatting')
t = time.time()
cPickle.dump( scales, open(  OUTPUT_DIR+ "tubes.p", "wb" ), protocol=cPickle.HIGHEST_PROTOCOL )
# writeLog("write to file took %.2f seconds"% (time.time()-t)) # 14.753661871 0.0874960422516
print(time.time()-t)

# Error caught on file: v_RockClimbingIndoor_g01_c02
# Error caught on file: v_PlayingDaf_g08_c04

# Error caught on file: v_RockClimbingIndoor_g01_c02
# Error caught on file: v_ApplyEyeMakeup_g01_c01
# Error caught on file: v_PlayingDaf_g08_c04