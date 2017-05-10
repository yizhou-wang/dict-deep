# Spare action recognition
# Lingyu & Yizhou
# Loading videos
# Extract keypoints
# Compute local motion pattern



import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import stats
# img1 = cv2.imread('box.png',0)          # queryImage
# img2 = cv2.imread('box_in_scene.png',0) # trainImage


def readvideo(source):
	cam = cv2.VideoCapture(source)
	imglist=[]
	# If Camera Device is not opened, exit the program
	if not cam.isOpened():
		print "Video device or file couldn't be opened"
		exit()

	# retval, img = cam.read()
	# print type(img)
	# print "frame"
	# imglist.append(img)
	i = 0
	while True:
		# Retrieve an image and Display it.
		retval, img = cam.read()
		 
		#print type(retval)
		i = i + 1
		
		
		
		if not retval:
			#print "Cannot capture frame device"
			break
		
		#print type(img)
		#image = rgb2gray(img)
		imglist.append(img)
		print '%s %s %s'%(i,'th','frame')
		
		

	return imglist

def subseq(imglist, N_sq):
	# N_sq is the number of subsequences.
	#N_frame = len(imglist)/N_sq

	chunks = [imglist[x:x+N_sq] for x in range(0, len(imglist), N_sq)]
	#print len(chunks[0]) # number of subsequences
	#print imglist[0].shape
	#print chunks[0][1].shape # the second subsequence
	#print type(chunks[0])
	return chunks

# def orbkeypoint(img_seq):

# 	# Note by lingyu, for one seq . extract keypoint list
# 	kp_list_seq = []
# 	L = len(img_seq) # number of frames in one seq.
# 	orb = cv2.ORB_create()
# 	for i in range(0, L):
# 		current_image = img_seq[i]
# 		print '%s %s %s'%('original','dimension',current_image.shape)
# 		kp1, des1 = orb.detectAndCompute(current_image,None)
# 		# kp1 is list
# 		kp_list_seq.append(kp1)
# 	return kp_list_seq

def orbkeypoint(img_frame):

	orb = cv2.ORB_create()

	current_image = img_frame
	print '%s %s %s'%('original','dimension',current_image.shape)
	kp1, des1 = orb.detectAndCompute(current_image,None)
	# kp1 is list
	
	return kp1


	# imglist = img
	# # print type(imglist[0])
	# #print (imglist[0]).shape[0]
	# print '%s %s %s'%('original','dimension',(imglist[0]).shape)
	
	# #print imglist
	# # Initiate SIFT detector
def matching(imglist):
	orb = cv2.ORB_create()
	img1 = imglist[0]
	img2 = imglist[1]
	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	# print des1
	# print des2
	# des1, des2 are the feature descriptors of keypoint

	# point keypoint location.
	print kp1[0].pt

	print '%s %s %s'%('has',len(kp1),'keypoints')
	print '%s %s %s'%('has',len(kp2),'keypoints')
	
	#print type(kp1)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	#matches = sorted(matches, key = lambda x:x.distance)

	# Draw first 10 matches.
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50], None,flags=2)

	plt.imshow(img3),plt.show()
	

	

# def kplocation(kp_list_seq):
	
# 	kploc_seq = []
# 	# for each frame in the seq
# 	for f in range(0, len(kp_list_seq)):
# 		kploc_frame = []
# 		kp_frame = kp_list_seq[f]
# 		# for each point in one frame
# 		for i in range(0, len(kp_frame)):
# 			kp = kp_frame[i]
# 			kp_xy = kp.pt
# 			kploc_frame.append(kp_xy)
# 		kploc_seq.append(kploc_frame)
# 	return kploc_seq

def kplocation(kp_frame):
	
	kploc_frame = []
	# for each point in one frame
	for i in range(0, len(kp_frame)):
		kp = kp_frame[i]
		kp_xy = kp.pt
		# tuple list
		kploc_frame.append(kp_xy)
	
	return kploc_frame

def kppatchfuc(kploc_frame, sub_video_seq):
	# get patch around the keypoint.
	kppatch = []
	for i in range(0,len(kploc_frame)):
		kploc_pt = kploc_frame[i]
		patch_center = kploc_pt
		patch_size = 24 # Note by Lingyu, increasing patch size will increasing accuracy
		patch_x = patch_center[0] - patch_size/2
		patch_y = patch_center[1] - patch_size/2

		sub_patch = []
		for f in range(0, len(sub_video_seq)):
			frame = rgb2gray(sub_video_seq[f])
			#print frame.shape
			patch_pt = frame[int(patch_y):int(patch_y+patch_size), int(patch_x):int(patch_x+patch_size)]
			sub_patch.append(patch_pt)

			# if patch_pt.shape == (0,24):
			# 	print patch_center, patch_x,patch_y
			# 	exit()
			#print patch_pt.shape

		kppatch.append(sub_patch)

	

	return kppatch

def motionpattern(kppatch,N_sq):
	k = len(kppatch)
	for i in range(0,len(kppatch)):
		# for ith keypoint
		patch = kppatch[i]
		# for ith keypoint, we have N_sq frames, so the dimentsion of patch is 24*24*N_sq
		# We need to calculate central moments for this N_sq data which is 24*24
		#print patch[0].shape
		patch_array = np.zeros(shape=[24,24,N_sq])
		for m in range(0,N_sq):
			#print patch[m].shape
			patch_array[:,:,m] = patch[m]
		#print patch_array.shape
		feature = np.zeros(shape=[k,24,24,3])
		feature[i,:,:,0] = stats.moment(patch_array,moment=2,axis=2)

		feature[i,:,:,1] = stats.moment(patch_array,moment=3,axis=2)

		feature[i,:,:,2] = stats.moment(patch_array,moment=4,axis=2)

	feature = np.reshape(feature,(k,1728))
	print 'shape'
	print feature.shape
	scipy.io.savemat("./feature.mat", feature)
	return feature


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def centralmoment(imglist):
	img = imglist[0]

	image = rgb2gray(img) 
	feature = cv2.moments(image)
	print feature.shape

# def hog(imglist, seq_id):
# 	winSize = (128,64)
# 	blockSize = (16,16)
# 	blockStride = (8,8)
# 	cellSize = (8,8)
# 	nbins = 9
# 	derivAperture = 0
# 	winSigma = -1
# 	histogramNormType = 0
# 	L2HysThreshold = 2.0000000000000001e-01
# 	gammaCorrection = 0
# 	nlevels = 64
# 	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
#                                         histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
# 	winStride = (8,8)
# 	padding = (8,8)
# 	locations = ((10,20),)

# 	nInterval = 10
# 	BigCount = 1

# 	index = 1

# 	#index = 0

# 	FirstEntryFlag = False


# 	while(index < len(imglist)):
# 		#print len(imglist)
# 		hogCount = 0
# 		for i in range(index,(index + nInterval)):
			
# 			# imgPath = pathToVideoFile + "/0 (" + str(i) + ").jpg"

# 			# #READ THE IMAGE HERE
# 			# img = cv2.imread(imgPath)
# 			print i
# 			#print '%s %s %s'%('for',i,'th image')
# 			img = imglist[i-1]
# 			#print type(img)
			

			
# 			img = cv2.resize(img, (160, 120))
# 			h1 = hog.compute(img,winStride,padding,locations).T
# 			#print "Shape of HOG features is: ", h1.shape
# 			temp = np.copy(h1)
# 			#print "Shape of temp is: ", temp.shape

# 			if(hogCount == 0):
# 				hogTemp = np.zeros((nInterval, len(temp[0])))
#  				#print "Shape of hogTemp is: ", hogTemp.shape
# 				hogTemp[hogCount]= temp[0]
# 				if (FirstEntryFlag == False):
# 				    FirstHOGEntry = np.copy(temp)
# 				    FirstEntryFlag = True
# 			else:
# 				hogTemp[hogCount]= temp

# 			hogCount += 1
				
					

# 				#HOGPH = self.computePCA(hogTemp)
# 		index += nInterval		
# 		HOGPH = computeHOGPH(hogTemp, FirstHOGEntry)
		
			


# # print HOGPH
# 	# print type(HOGPH)
# 	# np.savetxt('/Users/lingyuzhang/Spring17/4HighDimension/finalproject/ActionLocalization_finalproj/Feature_seq/Descr_' + str(seq_id) + '.mat',HOGPH)
# 	# print HOGPH.shape
# 	return HOGPH

# def computeHOGPH(array, firstEntry):
# 	hogph = firstEntry
# 	#hogph = np.copy(array[0])
# 	for j in range(1,len(array)):
# 	    hogph += array[j-1] - array[j]

# 	return hogph			    

if __name__ == '__main__':

	imglist = readvideo("video.avi")
	#matching(imglist)



	N = len(imglist)
	print '%s %s %s'%('total',N,'frames')
	N_sq = N
	video_seq = subseq(imglist, N_sq)
	print '%s %s %s'%('videoseq',len(video_seq[0]),'frames')
	print '%s %s %s'%('total',len(video_seq),'segments')
	# # Whole keypoint list is kp_list
	# kp_list = []
	# # for each seq, extract keypoints.
	# for i in range(0, N_sq):
	# 	img_seq = video_seq[0][i]
	# 	kp_list_seq =orbkeypoint(img_seq)
	# 	kp_list.append(kp_list_seq)

	# kp_frame = orbkeypoint(imglist[0])
	# kploc_frame = kplocation(kp_frame)



	# Extract patch for all the subsequence
	patch = []
	#video_feature = np.zeros(shape=[])
	for i in range(0, N/N_sq):
		sub_video_seq = video_seq[i]
		kp_frame = orbkeypoint(sub_video_seq[0])
		kploc_frame = kplocation(kp_frame)
		kppatch = kppatchfuc(kploc_frame, sub_video_seq)
		motionpattern(kppatch,N_sq)
		
		# feature = hog(sub_video_seq, i)
		#np.savetxt('/Users/lingyuzhang/Spring17/4HighDimension/finalproject/ActionLocalization_finalproj/Feature_seq/Descr_' + str(i) + '.mat',feature)
	

		#Extract patch for all the other frames in each subsequence
		
	# 	sub_patch_seq = []

	# 	# if i == 0:
	# 	# 	start = 1
	# 	# else:
	# 	# 	start = 0

	# 	for f in range(0, N_sq):
	# 		img_frame = sub_video_seq[f]
	# 		patch_frame = kppatch(kploc_frame, img_frame)
	# 		sub_patch_seq.append(patch_frame)
		
	# 	patch.append(sub_patch_seq)
	# # print len(patch)
	# # get feature for each subsequence
	# feature = []
	# for i in range(0, N/N_sq):
	# 	video_seq[i]
	# 	patch_gray_seq = []
	# 	#print type(patch[i])
	# 	for f in range(0, len(patch[i])):
	# 		patch_gray_frame = []
	# 		for psingle in range(0, len(patch[i][f])):
	# 			patch_frame_gray = rgb2gray((patch[i])[f][psingle])
	# 			patch_gray_frame.append(patch_frame_gray)
	# 		patch_gray_seq.append(patch_gray_frame)

	# 	feature_seq = motionpattern(patch_gray_seq)

	# # hog(imglist)
	# # 	# print len(sub_video_seq)
	# # 	feature_seq = motionpattern(patch, sub_video_seq, kp_frame, N_sq, N/N_sq)
	# # 	feature.append(feature_seq)
	# # centralmoment(imglist)
	# # subseq(imglist,2)
	# # kppatch(0)





