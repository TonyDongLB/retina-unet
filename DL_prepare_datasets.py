# -*- coding:utf-8 -*-  


#==========================================================
#
#  This prepare the hdf5 datasets of the  database
#
#============================================================

import os
import ConfigParser
import h5py
import numpy as np
from PIL import Image
import cv2

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./DL_datesets/training/images/"
groundTruth_imgs_train = "./DL_datesets/training/groundTruth/"
borderMasks_imgs_train = "./DL_datesets/training/mask/"
#test
original_imgs_test = "./DL_datesets/test/images/"
groundTruth_imgs_test = "./DL_datesets/test/groundTruth/"
borderMasks_imgs_test = "./DL_datesets/test/mask/"
#---------------------------------------------------------------------------------------------

config = ConfigParser.RawConfigParser()
config.read('DL_configuration.txt')

path_local = config.get('data paths', 'path_local')
train_imgs_original = config.get('data paths', 'train_imgs_original')
train_groundTruth = config.get('data paths', 'train_groundTruth')
train_border_masks = config.get('data paths', 'train_border_masks')
test_imgs_original = config.get('data paths', 'test_imgs_original')
test_groundTruth = config.get('data paths', 'test_groundTruth')
test_border_masks = config.get('data paths', 'test_border_masks')

Nimgs_test = 60
Nimgs_traning = 27
channels = 3
height = 1152
width = 1500
dataset_path = "./DL_datasets_training_testing/"



def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,Nimgs,train_test="null"):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    print imgs_dir
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            if files[i] == '.DS_Store':
                continue
            print "original image: " +files[i]
            img = Image.open(imgs_dir+files[i])

            # =====================================
            # 如果要可以重新使用cv2显示图片，使用如下操作
            # r, g, b = cv2.split(img)
            # b = b / 255.
            # g = g / 255.
            # r = r / 255.
            # img = cv2.merge([b, g, r])
            # =====================================

            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[i]
            print "ground truth name: " + groundTruth_name
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            #corresponding border masks
            border_masks_name = files[i]
            print "border masks name: " + border_masks_name
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)

    print "imgs max: " +str(np.max(imgs))
    print "imgs min: " +str(np.min(imgs))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print "ground truth and border masks are correctly withih pixel value range 0-255 (black-white)"
    #reshaping for my standard tensors
    # ATTENTION !!!! the dim!!!
    # imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,height,width,channels))
    groundTruth = np.reshape(groundTruth,(Nimgs,height,width,1))
    border_masks = np.reshape(border_masks,(Nimgs,height,width,1))
    assert(groundTruth.shape == (Nimgs,height,width,1))
    assert(border_masks.shape == (Nimgs,height,width,1))
    return imgs, groundTruth, border_masks


#getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,Nimgs_traning,"train")
print "saving train datasets"
write_hdf5(imgs_train, dataset_path + train_imgs_original)
write_hdf5(groundTruth_train, dataset_path + train_groundTruth)
write_hdf5(border_masks_train,dataset_path + train_border_masks)

#getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,Nimgs_test,"test")
print "saving test datasets"
write_hdf5(imgs_test,dataset_path + test_imgs_original)
write_hdf5(groundTruth_test, dataset_path + test_groundTruth)
write_hdf5(border_masks_test,dataset_path + test_border_masks)

print imgs_train.shape
print imgs_test.shape