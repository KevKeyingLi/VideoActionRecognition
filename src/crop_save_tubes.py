import cv2 
import numpy
import cPickle
import time

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

def crop_frames(video_path,start,end,avgsel,img_output_dir):
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
		elif frame_idx > end:
			break
		frame_idx += 1
	cap.release()
	# if not exists, create!
	# save the image files and return the cropped list of the video.
	return cropped_list


# video directory: '/data/UCF/data/Thumos/Videos/UCF101'
# 92612999952032581035060556



tube_file = '/data/UCF/data/Thumos/iDTF/Keying/output/tubes.p'
t = time.time()
scales = cPickle.load(open(tube_file,'rb'))
print(time.time()- t) # 157.41545105s
video_dir = '/data/UCF/data/Thumos/Videos/UCF101/'
img_output_dir = '/data/UCF/data/Thumos/iDTF/Keying/cropped_tubes/'
for key in scales[0]:
	print(key)
	for i in range(2):
		tubes = scales[i][key]
		print('Video '+key+' has '+str(len(tubes))+' tubes.')
		j = 0
		for tube in tubes:
			j+= 1
			start = tube['start']
			end = tube['end']# inclusive
			avgsel = tube['avgsel']
			frames = crop_frames(video_dir+key+'.avi',start,end,avgsel,img_output_dir+'scale_'+str(i)+'/'+str(key)+'/'+str(j)+'/')

		# out put two scales into different directories.



