import sys
import os
import numpy as np
import cv2
import heapq
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import math
from metric_learn import LMNN
import time


def chi_square_distance(featuresA, featuresB, eps=1e-10):

	dis = 0.5 * np.sum([((a - b) ** 2) for a, b in zip(featuresA, featuresB)])

	return dis

def cosine_distance(featuresA, featuresB):

	normA = np.linalg.norm(featuresA)
	normB = np.linalg.norm(featuresB)
	if normA == 0:
		normA = 1
	if normB == 0:
		normB = 1
	dis = 1 - np.dot(featuresA, featuresB)/(normA * normB)
	print "norm:", np.linalg.norm(featuresA), np.linalg.norm(featuresB)

	return dis

def l1_distance(featuresA, featuresB):

	dis = np.sum([np.absolute(a-b) for a, b in zip(featuresA, featuresB)])/np.sum([max(np.absolute(a), np.absolute(b)) for a, b in zip(featuresA, featuresB)])

	return dis
def covariance_matrix(Data):

	N = len(Data)
	#matrix_color
	dim = len(Data[0][0])
	mean_vector = np.zeros(dim)
	for data in Data:
		for i in xrange(len(data[0])):
			mean_vector[i] += data[0][i]
	mean_vector /= N
	matrix_color = np.zeros((dim, dim))
	for i in xrange(dim):
		for j in xrange(dim):
			for n in xrange(N):
				matrix_color[i, j] += (Data[n][0][i] - mean_vector[i]) * (Data[n][0][j] - mean_vector[j])
			matrix_color[i, j] /= N

	#matrix_shape
	dim = len(Data[0][1])
	mean_vector = np.zeros(dim)
	for data in Data:
		for i in xrange(len(data[1])):
			mean_vector[i] += data[1][i]
	mean_vector /= N
	matrix_shape = np.zeros((dim, dim))
	for i in xrange(dim):
		for j in xrange(dim):
			for n in xrange(N):
				matrix_shape[i, j] += (Data[n][1][i] - mean_vector[i]) * (Data[n][1][j] - mean_vector[j])
			matrix_shape[i, j] /= N


	return matrix_color, matrix_shape

def mahalanobis_metric(featuresA, featuresB, covari_matrix):

	X = np.array(featuresA) - np.array(featuresB)
	dis = float(np.dot(np.dot(X, covari_matrix), X.T))
	return dis



def histogram(image):
	
	hist = cv2.calcHist([image], channels=[0, 1, 2], mask=None, histSize=(3, 8, 8), ranges=[0, 180, 0, 256, 0, 256])
	
	hist = cv2.normalize(hist, dst=None).flatten()

	return hist

def getColorFeatures(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	features = []
	(h, w) = image.shape[:2]

	#split image to segments
	xseg_num = 4
	yseg_num = 4
	xseg = []
	yseg = []
	seg = []
	for i in xrange(0, xseg_num+1):
		if h - h/xseg_num * i <= 0:
			xseg.insert(0, 0)
		else:
			xseg.insert(0, h - h/xseg_num * i)
	for j in xrange(0, yseg_num+1):
		if w - w/yseg_num * j <= 0:
			yseg.insert(0, 0)
		else:
			yseg.insert(0, w - w/yseg_num * j)
	for i in xrange(len(xseg)-1):
		for j in xrange(len(yseg)-1):
			seg += [(xseg[i], xseg[i+1], yseg[j], yseg[j+1])]
	#for each seqment: compute histogram
	for (startX, endX, startY, endY) in seg:
		hist = histogram(image[startX:endX,startY:endY])
		features.extend(hist)
	
	return features

def getShapeFeatures(input_image):
	features = []
	image = cv2.resize(input_image,(128, 128))

	#winSize = (64,64)
	winSize = (64,64)
	blockSize = (16,16)
	blockStride = (8,8)
	cellSize = (8,8)
	nbins = 9
	derivAperture = 1
	winSigma = 4.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma, \
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
	#compute(img[, winStride[, padding[, locations]]]) -> descriptors
	winStride = (8,8)
	padding = (8,8)
	locations = ((10,20),)
	hist = hog.compute(image,winStride,padding,locations)
	hist = cv2.normalize(hist, dst=None).flatten()
	
	features.extend(hist)
	print >> sys.stderr, "features len:",len(features)
	return features


def getFeatures(imagePath):

	image = cv2.imread(imagePath)


	#color
	color_features = getColorFeatures(image)
	
	#texture
	#shape
	shape_features = getShapeFeatures(image)
	#fusion: use concat


	

	#return features
	#return color_features, shape_features
	return color_features, shape_features


def loadData(ImagesPath):


	#features = getfeatures(image)
	Data = []
	
	Truth = {}
	index = 0
	mapping = {}
	remapping = {}
	for i, (dirPath, dirNames, fileNames) in enumerate(os.walk(ImagesPath)):
		if i == 0:
			Labels = dirNames
		else:
			count = 0 #max = 20
			for f in fileNames:
				if count >= 20:
					break
				Truth[f] = Labels[i-1]
				imagepath =  os.path.join(dirPath, f)
				mapping[f] = len(Data)
				remapping[len(Data)] = f
				features = getFeatures(imagepath)
				Data.append(features)
				count += 1

	return Data, Truth, mapping, remapping

def loadQuery(QueryPath):

	Query = []
	for dirPath, dirNames, fileNames in os.walk(QueryPath):
		for f in fileNames:
			imagepath = os.path.join(dirPath, f)
			label = f.split(".")[0]
			features = getFeatures(imagepath)
			Query.append((label, features))
	return Query

def MetricLearning(Data, mapping, remapping, Truth):

	#build metric learing
	Truth_Label = list(set(Truth.values()))

	X_color = np.array([data[0] for data in Data])
	#print >> sys.stderr, "X_color concat X_shape", X_color.shape
	#X_color = np.array([data[0] for data in Data])
	Y_color = np.array([Truth_Label.index(Truth[remapping[i]]) for i, data in enumerate(Data)])

	
	print >> sys.stderr, "X_color", X_color.shape, Y_color.shape
	

	s = time.time()

	lmnn_color = LMNN(k=5, min_iter=0, max_iter=400, learn_rate=1e-6)
	lmnn_color.fit(X_color, Y_color, verbose=True)

	print >> sys.stderr, time.time() - s, "color learning done"

	'''X_shape = np.array([data[1] for data in Data])
	Y_shape = np.array([Truth_Label.index(Truth[remapping[i]]) for i, data in enumerate(Data)])
	print >> sys.stderr, "X_shape", X_shape.shape, Y_shape.shape

	s = time.time()

	lmnn_shape = LMNN(k=20, min_iter=0, max_iter=1, learn_rate=1e-6)
	lmnn_shape.fit(X_shape, Y_shape, verbose=True)

	print >> sys.stderr, time.time() - s, "shape learning done"
	return lmnn_color, lmnn_shape'''
	return lmnn_color


def getList(query, Data, index,mapping, remapping, lmnn_color):


	co_matrix_color = lmnn_color.metric()
	#co_matrix_shape = lmnn_shape.metric()
	
	#brute force search~
	
	N = len(Data) # leave-one-out
	# dim = 2 = (color, shape)
	dim = 2
	mean_vector = np.zeros(2)

	#Data_reRank = []
	h = []
	for i, data in enumerate(Data):

		if i == index:
			continue


		#dis = l1_distance(query[1][0], data[0])
		#dis = l1_distance(query[1][1], data[1])
		#dis_shape = l1_distance(query[1][1], data[1])
		#dis_color = cosine_distance(query[1][0], data[0])
		#dis_shape = cosine_distance(query[1][1], data[1])
		dis = mahalanobis_metric(query[1][0], data[0], co_matrix_color)
		#dis_shape = mahalanobis_metric(query[1][1], data[1], co_matrix_shape)

		heapq.heappush(h, (dis, remapping[i]))
		#Data_reRank.append([dis_color, dis_shape])
		#mean_vector[0] += dis_color
		#mean_vector[1] += dis_shape

	similist = [heapq.heappop(h) for i in range(len(h))]
	co_matrix = np.zeros((dim, dim))
	'''
	mean_vector /= N
	co_matrix = np.zeros((dim, dim))
	for i in xrange(dim):
		for j in xrange(dim):
			for n in xrange(N):
				co_matrix[i, j] += (Data_reRank[n][i] - mean_vector[i]) * (Data_reRank[n][j] - mean_vector[j])
			co_matrix[i, j] /= N
	


	Lambda = 1.
	h = []
	for i, data in enumerate(Data_reRank):
		if i == index:
			continue
		dis = (data[0] - mean_vector[0])/co_matrix[0,0] + Lambda* (data[1] - mean_vector[1])/co_matrix[1,1]
		heapq.heappush(h, (dis, remapping[i]))
	'''

	


	return similist, co_matrix, mean_vector

def computeAP(queryType, similist, Truth):

	y_true = np.array([Truth[s[1]]==queryType for s in similist])
	y_scores = np.array([-s[0] for s in similist])
	AP = average_precision_score(y_true, y_scores) 
	precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
	print len(precision)
	print len(recall)

	return AP, y_true, y_scores

if __name__ == "__main__":



	#------------load data-----------#

	ImagesPath = sys.argv[1]
	ADDImagesPath = sys.argv[2]

	Data, Truth, mapping, remapping = loadData(ImagesPath)
	Data_ADD, Truth_ADD, mapping_ADD, remapping_ADD = loadData(ADDImagesPath)
	print >> sys.stderr, "loading data completed"

	#------------build model---------#

	lmnn_color = MetricLearning(Data_ADD, mapping_ADD, remapping_ADD, Truth_ADD)
	#lmnn_color, lmnn_shape = covariance_matrix(Data)
	print >> sys.stderr, "modeling completed"

	#------------evaluation----------#

	#QueryPath = sys.argv[2]

	#Query = loadQuery(QueryPath)
	#print >> sys.stderr, "loading query completed"

	MAP = 0.
	mAP = 0.
	Y_TRUE = []
	Y_SCORES = []
	#symbols = ['-', '--', '-.', ':', '.', ',', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', '+', 'x', 'D', '1', '2']
	for i, data in enumerate(Data):
		query = (Truth[remapping[i]], data)
		print >> sys.stderr, "query", query[0]
		
		
		similist, co_matrix, mean_vector = getList(query, Data, i, mapping, remapping,lmnn_color)
		print(query[0].center(40, "="))
		for s in similist:
			print s
		AP, y_true, y_scores = computeAP(query[0], similist, Truth)
		print "AP:", AP
		mAP += AP
		if i % 20 == 19:
			print "mAP:", mAP/20
			MAP += mAP/20
			mAP = 0.

		Y_TRUE.extend(y_true)
		Y_SCORES.extend(y_scores)
		#Precision += precision
		#Recall += recall
		

		#plt.plot(precision, recall, symbols[0], label=query[0])
		#symbols.pop(0)


		
		#plt.savefig("./plot/"+query[0]+".png")
		#plt.clf()
		

	#precision, recall, _ = precision_recall_curve(Y_TRUE, Y_SCORES)

	MAP /= 20
	
	print "MAP:", MAP

	

	#plt.scatter(Precision, Recall)
	#plt.show()
	





