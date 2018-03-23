import numpy as np
import random
import ConfigParser
import cv2
import os
import cv

from DL_help_functions import load_hdf5
from DL_help_functions import visualize
from DL_help_functions import group_images
from DL_help_functions import strelDisk

from DL_pre_processing import my_PreProc

#====================== test the picture is right==================
train_imgs_original = load_hdf5('../DL_datasets_training_testing/' + 'DL_dataset_imgs_train.hdf5')
train_mask = load_hdf5('../DL_datasets_training_testing/' + 'DL_dataset_borderMasks_train.hdf5')
train_GD = load_hdf5('../DL_datasets_training_testing/' + 'DL_dataset_groundTruth_train.hdf5')
train_imgs = train_imgs_original
# train_imgs = my_PreProc(train_imgs_original)
print train_imgs.shape
for i in range(27):
	img = train_imgs[i]
	mask = train_mask[i]
	GD = train_GD[i]
	img = np.array(img, dtype= np.uint8)
	mask = np.array(mask, dtype= np.uint8)
	r, g, b = cv2.split(img)
	img = cv2.merge([b, g, r])
	print str(mask.shape) + 'mask.shape'

	img = cv2.bitwise_and(img, img, mask= mask)
	avgL = 70.
	Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(Lab)
	# print np.sum(l)
	# print np.sum(mask)
	imgAvgL = np.sum(l) / (np.sum(mask) / 255.)
	print 'imgAvgL' + str(imgAvgL)
	if imgAvgL < avgL:
		l += int(avgL - imgAvgL)
	Lab = cv2.merge([l, a, b])
	Lab = cv2.bitwise_and(Lab, Lab, mask= mask)
	img = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
	img = cv2.bitwise_and(img, img, mask= mask)

	# img = cv2.medianBlur(img, 3)

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

	b, g, r = cv2.split(img)
	g = clahe.apply(g)
    #
	# Iwr = np.zeros(img.shape, dtype= np.uint8)
	# Iwd = np.zeros(img.shape, dtype= np.uint8)
	# Ibr = np.zeros(img.shape, dtype= np.uint8)
	# Ibd = np.zeros(img.shape, dtype= np.uint8)
    #
    #
	# Iwrchannels = cv2.split(Iwr)
	# Iwdchannels = cv2.split(Iwd)
	# Ibrchannels = cv2.split(Ibr)
	# Ibdchannels = cv2.split(Ibd)
    #
	# for j in range(3, 12):
    #
	# 	B = strelDisk(j)
	# 	BPlus = strelDisk(j + 1)
    #
	# 	temp1 = cv2.morphologyEx(r, cv2.MORPH_TOPHAT, B)
	# 	Iwrchannels[2] = cv2.max(Iwrchannels[2], temp1)
	# 	temp1 = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, B)
	# 	Iwrchannels[1] = cv2.max(Iwrchannels[1], temp1)
	# 	temp1 = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, B)
	# 	Iwrchannels[0] = cv2.max(Iwrchannels[0], temp1)
    #
	# 	temp1 = cv2.morphologyEx(r, cv2.MORPH_OPEN, B)
	# 	temp2 = cv2.morphologyEx(r, cv2.MORPH_OPEN, BPlus)
	# 	Iwdchannels[2] = cv2.max(Iwdchannels[2], temp1 - temp2)
	# 	temp1 = cv2.morphologyEx(g, cv2.MORPH_OPEN, B)
	# 	temp2 = cv2.morphologyEx(g, cv2.MORPH_OPEN, BPlus)
	# 	Iwdchannels[1] = cv2.max(Iwdchannels[1], temp1 - temp2)
	# 	temp1 = cv2.morphologyEx(b, cv2.MORPH_OPEN, B)
	# 	temp2 = cv2.morphologyEx(b, cv2.MORPH_OPEN, BPlus)
	# 	Iwdchannels[0] = cv2.max(Iwdchannels[0], temp1 - temp2)
    #
	# 	temp1 = cv2.morphologyEx(r, cv2.MORPH_BLACKHAT, B)
	# 	Ibrchannels[2] = cv2.max(Ibrchannels[2], temp1)
	# 	temp1 = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, B)
	# 	Ibrchannels[1] = cv2.max(Ibrchannels[1], temp1)
	# 	temp1 = cv2.morphologyEx(b, cv2.MORPH_BLACKHAT, B)
	# 	Ibrchannels[0] = cv2.max(Ibrchannels[0], temp1)
    #
	# 	temp1 = cv2.morphologyEx(r, cv2.MORPH_CLOSE, B)
	# 	temp2 = cv2.morphologyEx(r, cv2.MORPH_CLOSE, BPlus)
	# 	Ibdchannels[2] = cv2.max(Ibdchannels[2], temp2 - temp1)
	# 	temp1 = cv2.morphologyEx(g, cv2.MORPH_CLOSE, B)
	# 	temp2 = cv2.morphologyEx(g, cv2.MORPH_CLOSE, BPlus)
	# 	Ibdchannels[1] = cv2.max(Ibdchannels[1], temp2 - temp1)
	# 	temp1 = cv2.morphologyEx(b, cv2.MORPH_CLOSE, B)
	# 	temp2 = cv2.morphologyEx(b, cv2.MORPH_CLOSE, BPlus)
	# 	Ibdchannels[0] = cv2.max(Ibdchannels[0], temp2 - temp1)
    #
	# Iwr = cv2.merge(Iwrchannels)
	# Iwd = cv2.merge(Iwdchannels)
	# Ibr = cv2.merge(Ibrchannels)
	# Ibd = cv2.merge(Ibdchannels)
    #
	# img = img + Iwr + Iwd - Ibr - Ibd
	# img = cv2.bitwise_and(img, img, mask= mask)





	# c1, c2, c3 = cv2.split(a)
	# c1 = c1 / 255.
	# c2 = c2 / 255.
	# c3 = c3 / 255.
	# r = c1
	# g = c2
	# b = c3

	# r = np.array(r, dtype = np.uint8)
	# b = np.array(b, dtype = np.uint8)
	# g = np.array(g, dtype = np.uint8)

	# r = np.array(r, dtype = np.float64)
	# b = np.array(b, dtype = np.float64)
	# g = np.array(g, dtype = np.float64)
	# r = np.array(r, dtype = np.uint8)
	# b = np.array(b, dtype = np.uint8)
	# g = np.array(g, dtype = np.uint8)
	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	# g = clahe.apply(g)


	# a = cv2.merge([b, g, r])
	print a.shape
	# a = np.transpose(a, (1, 2, 0))
	print a.shape
	# cv2.imshow('c1', c1)
	# cv2.imshow('c2', c2)
	# cv2.imshow('c3', c3)
	# cv2.imshow('mask', mask)
	cv2.imshow(str(i), g)
	cv2.imshow(str(i) + 'GD', GD)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
#====================== test the picture is right==================



# #====================== make the mask   ==================
# train_file = open('/Users/apple/Desktop/using/testset.txt')
# for img in train_file:
# 	nowImg = cv2.imread('/Users/apple/Desktop/using/ddb1_fundusimages/' + img[: -1])
# 	print nowImg.shape
# 	# nowImg = cv2.resize(nowImg, (584, 566))
# 	# nowImg = cv2.GaussianBlur(nowImg, (7, 7), 0)
# 	nowImg = cv2.medianBlur(nowImg, 13)



# 	nowGT = cv2.imread('/Users/apple/Desktop/using/EXGT/' + img[: -1], 0)
# 	b, g, r = cv2.split(nowImg)
# 	lab = cv2.cvtColor(nowImg, cv2.COLOR_BGR2LAB)
# 	l, a, b = cv2.split(lab)
# 	# cv2.imshow('r', r)
# 	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# 	l = clahe.apply(l)
# 	# cv2.waitKey(0)

# 	junk, mask = cv2.threshold(l, 10, 255, cv2.THRESH_BINARY)
# 	# kernel = np.ones((7, 7), np.uint8)
# 	# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# 	cv2.imwrite('/Users/apple/Documents/NN_Models/my_Unet/DL_datesets/test/images/'+ img[: -1], nowImg)
# 	cv2.imwrite('/Users/apple/Documents/NN_Models/my_Unet/DL_datesets/test/mask/'+ img[: -1], mask)
# 	cv2.imwrite('/Users/apple/Documents/NN_Models/my_Unet/DL_datesets/test/groundTruth/'+ img[: -1], nowGT)
# #====================== make the mask   ==================

# train_file = open('/Users/apple/Desktop/using/trainset.txt')


print 'yes'