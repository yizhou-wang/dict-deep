# Spare action recognition
# Lingyu & Yizhou
# Loading videos
# Extract keypoints
# Compute local motion pattern


import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from scipy import stats

import os
from os import listdir
from os.path import isfile, join

import cv2


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
			# print "Cannot capture frame device"
			break
		
		#print type(img)
		#image = rgb2gray(img)
		imglist.append(img)
		# print '%s %s %s'%(i,'th','frame')
		
		

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


def orbkeypoint(img_frame):

	orb = cv2.ORB()

	current_image = img_frame
	print '%s %s %s'%('original','dimension',current_image.shape)
	kp1, des1 = orb.detectAndCompute(current_image,None)
	# kp1 is list
	
	return kp1


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
	patch_array = np.zeros(shape=[24,24,N_sq])
	feature = np.zeros(shape=[k,24,24,3])
	for i in range(0,len(kppatch)):
		# for ith keypoint
		patch = kppatch[i]
		# for ith keypoint, we have N_sq frames, so the dimentsion of patch is 24*24*N_sq
		# We need to calculate central moments for this N_sq data which is 24*24
		#print patch[0].shape
		
		for m in range(0,N_sq):
			#print patch[m].shape
			patch_array[:,:,m] = patch[m]
		#print patch_array.shape
		
		feature[i,:,:,0] = stats.moment(patch_array,moment=2,axis=2)
		feature[i,:,:,1] = stats.moment(patch_array,moment=3,axis=2)
		feature[i,:,:,2] = stats.moment(patch_array,moment=4,axis=2)

	feature = np.reshape(feature,(k,1728))
	print 'shape'
	print feature.shape
	
	return feature


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# def centralmoment(imglist):
# 	img = imglist[0]

# 	image = rgb2gray(img) 
# 	feature = cv2.moments(image)
# 	print feature.shape




if __name__ == '__main__':

	dataset_name = 'weizmann'
	root_dir = '../dataset/' + dataset_name + '/'

	
	cat_folders = [f for f in listdir(root_dir) if not isfile(join(root_dir, f))]
	print cat_folders

	for cat in cat_folders:
		
		cat_dir = root_dir + cat + '/'
		print cat_dir

		video_dirs = []
		name_list = sorted(os.listdir(cat_dir))
		video_name_list = []
		for video in name_list:
			if video.endswith(".avi"):
				video_name_list.append(video)

		# For training videos
		for video in video_name_list[:-1]:
			print 'Working on', video
			video_name = video.split('.')[0]
			cur_video_dir = os.path.join(cat_dir, video)
			video_dirs.append(cur_video_dir)

			imglist = readvideo(cur_video_dir)
			print video, 'Read!'
			#matching(imglist)


			N = len(imglist)
			print '%s %s %s'%('total',N,'frames')
			N_sq = N
			video_seq = subseq(imglist, N_sq)
			print '%s %s %s'%('videoseq',len(video_seq[0]),'frames')
			print '%s %s %s'%('total',len(video_seq),'segments')


			# Extract patch for all the subsequence
			patch = []
			for i in range(0, N / N_sq):
				sub_video_seq = video_seq[i]
				kp_frame = orbkeypoint(sub_video_seq[0])
				kploc_frame = kplocation(kp_frame)
				kppatch = kppatchfuc(kploc_frame, sub_video_seq)
				feature = motionpattern(kppatch,N_sq)

				mat_dir = '../results/' + dataset_name + '_features/'
				if not os.path.exists(mat_dir):
					os.makedirs(mat_dir)
				mat_name = mat_dir + video_name + '_train.mat'
				scipy.io.savemat(mat_name, {"feature": feature})

				print 'MAT:', mat_name, 'saved!'
		
		# For test video
		video = video_name_list[-1]
		print 'Working on', video
		video_name = video.split('.')[0]
		cur_video_dir = os.path.join(cat_dir, video)
		video_dirs.append(cur_video_dir)

		imglist = readvideo(cur_video_dir)
		print video, 'Read!'
		#matching(imglist)


		N = len(imglist)
		print '%s %s %s'%('total',N,'frames')
		N_sq = N
		video_seq = subseq(imglist, N_sq)
		print '%s %s %s'%('videoseq',len(video_seq[0]),'frames')
		print '%s %s %s'%('total',len(video_seq),'segments')


		# Extract patch for all the subsequence
		patch = []
		for i in range(0, N / N_sq):
			sub_video_seq = video_seq[i]
			kp_frame = orbkeypoint(sub_video_seq[0])
			kploc_frame = kplocation(kp_frame)
			kppatch = kppatchfuc(kploc_frame, sub_video_seq)
			feature = motionpattern(kppatch,N_sq)

			mat_dir = '../results/' + dataset_name + '_features/'
			if not os.path.exists(mat_dir):
				os.makedirs(mat_dir)
			mat_name = mat_dir + video_name + '_test.mat'
			scipy.io.savemat(mat_name, {"feature": feature})

			print 'MAT:', mat_name, 'saved!'




