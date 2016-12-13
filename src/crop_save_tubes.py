import cv2 
import numpy as np
import cPickle
import time
import os
import datetime


def writeLog(msg):
	msg = str(datetime.datetime.now()) + ':   ' + msg;
	print(msg)
	if not os.path.exists(os.path.dirname(logFileLoc)):
		os.makedirs(os.path.dirname(logFileLoc))
	with open(logFileLoc, 'a') as logFile:
		logFile.write( msg+'\n')


def save_image(file_str, matrix):
	if not os.path.exists(os.path.dirname(file_str)):
		os.makedirs(os.path.dirname(file_str))
	cv2.imwrite(file_str,matrix)

def crop_frames(video_path,start,end,avgsel,img_output_dir,image_list):
	# 41.0000  124.5000  221.7500  276.0000 # 240*320
	cap = cv2.VideoCapture(video_path)
	frame_idx = 1
	cropped_list = []
	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret:
			print('ret not true')
			break
		if frame_idx >= start and frame_idx <= end:
			# print(avgsel[frame_idx-1][0])
			y1,x1,y2,x2 = avgsel[frame_idx-1][0]
			y1 = int(y1)
			x1 = int(x1)
			y2 = int(y2)
			x2 = int(x2)
			# print([y1,x1,y2,x2])
			cropped_frame = frame[y1-1:y2,x1-1:x2]
			# print([y1-1,y2,x1-1,x2])
			cropped_list.append(cropped_frame)
			save_image(img_output_dir+str(frame_idx)+'.png',cropped_frame)
			image_list.append(img_output_dir+str(frame_idx)+'.png')
		elif frame_idx > end:
			break
		frame_idx += 1
	cap.release()
	# if not exists, create!
	# save the image files and return the cropped list of the video.
	return cropped_list


# video directory: '/data/UCF/data/Thumos/Videos/UCF101'
# 92612999952032581035060556

logFileLoc = '/data/UCF/data/Thumos/iDTF/Keying/cropped_tubes/crop_log.log'
tube_file = '/data/UCF/data/Thumos/iDTF/Keying/output/tubes.p'
writeLog('Start loading the tubes from pickle file')
t = time.time()
scales = cPickle.load(open(tube_file,'rb'))
writeLog('Finished loading the tubes after %.2f seconds' %(time.time()- t) )# 157.41545105s
video_dir = '/data/UCF/data/Thumos/Videos/UCF101/'
img_output_dir = '/data/UCF/data/Thumos/iDTF/Keying/cropped_tubes/'
img_lists = list([list(),list()])
total = len(scales[0])
cnt = 0
for key in scales[0]:
	print(key)
	for i in range(2):
		tubes = scales[i][key]
		print('Video '+key+' scale '+str(i)+' has '+str(len(tubes))+' tubes.')
		j = 0
		for tube in tubes:
			j+= 1
			print('\t Tube %d' %j)
			start = tube['start']
			end = tube['end']# inclusive
			avgsel = tube['avgsel']
			frames = crop_frames(video_dir+key+'.avi',start,end,avgsel,img_output_dir+'scale_'+str(i)+'/'+str(key)+'/'+str(j)+'/',img_lists[i])
	cnt +=1
	print('finished %d/%d videos '%(cnt,total))
writeLog('Save the list of image files')
t = time.time()
cPickle.dump(img_lists, open(img_output_dir+'image_lists.p','wb'), protocol=cPickle.HIGHEST_PROTOCOL )
writeLog('Finished dump lists of images after %.2f seconds' %(time.time()- t) )# 157.41545105s
writeLog('Program finished')
		# out put two scales into different directories.

# Next tasks:
#  Maybe maintain a list of images

