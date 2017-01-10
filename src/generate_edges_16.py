import numpy as np
import scipy.io as sio
def generate_edge_16(window_list,file_name):
	edge_mat = np.empty((len(window_list),len(window_list)))
	head = 0
	tail = 0
	i = 0
	cur_video_name  = ''
	for window in window_list:
		if cur_video_name == '':
			cur_video_name = window[0]
		elif cur_video_name != window[0]:
			tail = i#exclusive
			fill_matrix(edge_mat,head,10)
			head = i
		i += 1
	sio.savemat(file_name,{'edge_mat':edge_mat})
	return

def fill_matrix(edge_mat,start,end,max_dist):# end exclusive
	for i in range(start,end):
		for j in range(i, min(end,start+max_dist)):
			edge_mat[i,j] = max_dist-(j-i)
			edge_mat[j,i] = max_dist-(j-i)