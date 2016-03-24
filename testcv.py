import numpy as np
import cv2
import sys
import os
import math
def getImages(FrameDir, FrameNum):
	Images = []
	Orishape = tuple()
	for i in xrange(1, FrameNum+1):
		filename = FrameDir + str(i) +'.jpg'


		img = cv2.imread(filename)
		#hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		#img = hsv_img
		Orishape = img.shape
		Images.append(img)
		#cv2.imshow('image',img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

	print Orishape
	return Images, Orishape

def getHImages(Images):
	HImages = []
	for img in Images:
		img = img.reshape((img.shape[0]*img.shape[1]*img.shape[2]),order='F')
		hist0, bin = np.histogram(img[0:Orishape[0]*Orishape[1]], bins=np.arange(0,257,4))
		hist1, bin = np.histogram(img[Orishape[0]*Orishape[1]:2*Orishape[0]*Orishape[1]], bins=np.arange(0,257,4))
		hist2, bin = np.histogram(img[2*Orishape[0]*Orishape[1]:3*Orishape[0]*Orishape[1]], bins=np.arange(0,257,4))
		hist = np.array(list(hist0) + list(hist1) + list(hist2))
		HImages.append(hist)
	print len(HImages)
	print HImages[0].shape
	return HImages

def cosine_distance(hist1,hist2):
	if np.array_equal(hist1, hist2):
		return 0.0
	return 1 - np.dot(hist1,hist2)/(np.linalg.norm(hist1)*np.linalg.norm(hist2))

def getCuts(HImages, threshold):
	Cuts = [0]
	for i in xrange(1, len(HImages)):
		if cosine_distance(HImages[i-1],HImages[i]) > threshold:
			Cuts.append(i)
	
	return Cuts

def blockshaped(arr, nrows, ncols):
    '''
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    '''
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
def getHImages_region(Images):
	HImages = []
	for img in Images:
		#a region = 4*4
		#img = img.reshape((img.shape[0]*img.shape[1]*img.shape[2]),order='F')
		hist = []
		for k in xrange(img.shape[2]):
			'''
			for i in xrange(0, img.shape[0], 4):
				for j in xrange(0, img.shape[1], 4):
					tmphist = []
					for ii in xrange(i, i+4):
						for jj in xrange(j, j+4):
							tmphist.append(img[ii,jj,k])
					tmphist, bin = np.histogram(tmphist, bins=np.arange(0,257,4))
					
					hist += list(tmphist)'''
			blocks = blockshaped(img[:,:,k], 4, 4)
			for block in blocks:
				
				tmphist, bin = np.histogram(block, bins=np.arange(0,257,4))
				hist += list(tmphist)


		hist = np.array(hist)
		print hist.shape, len(HImages)
		HImages.append(hist)
	print len(HImages)
	print HImages[0].shape
	return HImages
def getCuts_region(HImages, threshold_histogram, threshold_count):
	Cuts = [0]
	for i in xrange(1, len(HImages)):
		count = 0
		for j in xrange(0, HImages[i].shape[0], 64):
			hist1 = HImages[i-1][j:j+64]
			hist2 = HImages[i][j:j+64]
			
			#print cosine_distance(hist1, hist2), count
			if cosine_distance(hist1, hist2) > threshold_histogram:
				count += 1
		if count > threshold_count*HImages[i].shape[0]/64:

			Cuts.append(i)
			#break
		#if i in range(92,115)+range(164,186)+range(235,257)+range(284,306)+range(350,386)+range(410,415)+range(445,450)+rang3([0, 68, 123, 177, 215, 328, 363, 420, 474, 581, 654, 708, 769]:
		print i, count, threshold_count*HImages[i].shape[0]/64
	
	return Cuts
def importance(key, cluster, Total_length):
	# long = important
	
	return key[1] * math.log(float(Total_length)/float(cluster['LENGTH']))
def hierarchical_clustering(HImages, Keys, threshold_hie_clustering):

	Clusters = []

	for key in Keys:
		#key = (index, lens)
		if len(Clusters) == 0:
			Clusters.append({'LIST':[key], 'SUM':HImages[key[0]], 'LENGTH':key[1]})
		else:
			flag = -1
			min_distance = 10
			close_cluster = -1
			for cluster in Clusters:
				if cosine_distance(cluster['SUM']/len(cluster['LIST']), HImages[key[0]]) < threshold_hie_clustering:
					
					if cosine_distance(cluster['SUM']/len(cluster['LIST']), HImages[key[0]]) < min_distance:
						min_distance = cosine_distance(cluster['SUM']/len(cluster['LIST']), HImages[key[0]])
						close_cluster = cluster
					flag = 1
			
			if flag == -1:
				#new cluster
				Clusters.append({'LIST':[key], 'SUM':HImages[key[0]], 'LENGTH':key[1]})
			else:
				close_cluster['LIST'].append(key)
				close_cluster['SUM'] += HImages[key[0]]
				close_cluster['LENGTH'] += key[1]



	return Clusters
if __name__ == "__main__":

	FrameDir = sys.argv[1]
	threshold_hie_clustering = float(sys.argv[2])
	threshold_histogram = float(sys.argv[3])
	try:
		threshold_count = float(sys.argv[4]) #Region Histogram
	except:
		threshold_count = -1 #Histogram
	FrameNum = len(os.listdir(FrameDir))
	Images, Orishape = getImages(FrameDir, FrameNum)

	if threshold_count == -1: #Histogram
		HImages = getHImages(Images)
		Cuts = getCuts(HImages, threshold_histogram)
	else:
		HImages = getHImages_region(Images)
		Cuts = getCuts_region(HImages, threshold_histogram, threshold_count)
	print Cuts
	#make key frames
	Keys = []

	for i in range(len(Cuts)):

		
		if i == len(Cuts) - 1:
			index = (len(HImages)+Cuts[i])/2
			img = Images[index]
			himg = HImages[index]
			Cuts[i] = len(HImages) - Cuts[i]
			lens = Cuts[i]
		else:
			index = (Cuts[i+1]+Cuts[i])/2
			img = Images[index]
			himg = HImages[index]
			Cuts[i] = Cuts[i+1] - Cuts[i]
			lens = Cuts[i]
		Keys.append((index,lens))
	print sum(Cuts)/float(len(Cuts))
	#print hist, hist.shape
	#cv2.imwrite('messigray.png',img)


	#here is a summarization algorithm
	Clusters = hierarchical_clustering(HImages, Keys, threshold_hie_clustering)
	print Clusters
	#use importance to define the size of key frames: if > 0.5*max -> size 1, else -> size 1/2
	key2cluster = {}
	MAX_IMPORT = 0
	for cluster in Clusters:

		if len(cluster['LIST']) == 1:
			continue
		else:
			max_import = 0
			max_key = -1
			for key in cluster['LIST']:
				if importance(key, cluster, len(Images)) > max_import:
					max_import = importance(key, cluster, len(Images))
					max_key = key
			cluster['LIST'] = [max_key]
	# select representation of cluster by importance

	#print Clusters
	for cluster in Clusters:
		for key in cluster['LIST']:
			key2cluster[key] = cluster
			if importance(key, cluster, len(Images)) > MAX_IMPORT:
				MAX_IMPORT = importance(key, cluster, len(Images))

	# set frame size by importance
	frames_size = {}
	for cluster in Clusters:
		for key in cluster['LIST']:
			if importance(key, cluster, len(Images)) > 0.5 *MAX_IMPORT:
				frames_size[key] = {'H':2, 'W':2}
			else:
				frames_size[key] = {'H':1, 'W':2}

	shaped_key_img = {}
	for key in frames_size:
		print key
		# reshape the images
		shaped_key_img[key[0]] = cv2.resize(Images[key[0]],  (Images[key[0]].shape[1], Images[key[0]].shape[0]*frames_size[key]['H']/2))
	
	#place the images
	W = 10
	H_row = 2
	H_row_top = 0
	A = np.zeros((H_row, W))
	S = np.zeros((A.shape[0]*120, A.shape[1]*160, 3))
	w = 0
	h = H_row_top
	prev_size = 0
	for key in Keys:
		if key in key2cluster:
			print key[0]
			print h, H_row_top
			size = frames_size[key]
			if h + size['H'] > H_row_top + H_row:
				#if a column is full, change to next column
				w += prev_size['W']
				h = H_row_top
			if w + size['W'] > W:
				#if a row is full, concat another row
				A = np.concatenate((A, np.zeros((H_row, W))), axis=0)
				S = np.concatenate((S, np.zeros((H_row*120, W*160, 3))), axis=0)
				w = 0
				h = H_row_top + H_row
				H_row_top += 2
			for i in xrange(size['H']):
				for j in xrange(size['W']):
					A[h+i, w+j] = key[0]
			for i in xrange(size['H']*120):
				for j in xrange(size['W']*160):
					#print (i,j), size['H']*120, shaped_key_img[key[0]].shape

					S[120*h+i, 160*w+j] = shaped_key_img[key[0]][i, j]
			h += size['H']
			prev_size = size

	print A


	#original shape = 240 * 320
	# output = 'out.jpg'
	cv2.imwrite('out.jpg', S)

	






















