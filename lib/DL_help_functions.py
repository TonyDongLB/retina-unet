import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

#convert RGB image in black and white
def rgb2green(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0. + rgb[:,1,:,:]*1. + rgb[:,2,:,:]*0.
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

#group a set of images row per columns
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg


#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img


#prepare the mask in the right shape for the Unet
def masks_Unet(masks):
    assert (len(masks.shape)==4)  #4D arrays
    assert (masks.shape[1]==1 )  #check the channel is 1
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,2))
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if  masks[i,j] == 0:
                new_masks[i,j,0]=1
                new_masks[i,j,1]=0
            else:
                new_masks[i,j,0]=0
                new_masks[i,j,1]=1
    return new_masks


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2]==2 )  #check the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1]
    elif mode=="threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print "mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'"
        exit()
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
    return pred_images

# Radius should be smaller than 22
def strelDisk(Radius):
    Radius = int(Radius)

    if Radius == 1:
        borderWidth = 1
        sel = np.ones((2 * Radius + 1, 2 * Radius + 1), dtype= np.uint8)
    else:
        if Radius == 2:
            borderWidth = 2
            sel = np.ones((2 * Radius + 1, 2 * Radius + 1), dtype=np.uint8)
        else:
            if Radius == 3:
                borderWidth = 0
                sel = np.ones((2 * Radius - 1, 2 * Radius - 1), dtype=np.uint8)
            else:
                n = Radius / 7
                m = Radius % 7
                if(m == 0 or m >= 4):
                    borderWidth = 2 * (2 * n + 1)
                else:
                    borderWidth = 2 * 2 * n
                sel = np.ones((2 * Radius - 1, 2 * Radius - 1), dtype=np.uint8)

    numZero = 0
    if Radius == 1:
        numZero = 1
    if Radius > 1 and Radius < 8:
        if Radius == 3:
            numZero = 0
        numZero = 2
    if Radius > 7 and Radius < 11:
        numZero = 4
    if Radius > 10 and Radius < 15:
        numZero = 6
    if Radius > 14 and Radius < 18:
        numZero = 8
    if Radius > 17 and Radius < 22:
        numZero = 10

    for i in range(sel.shape[0]):
        if (Radius - 1 - i) > (Radius - 1 - numZero):
            for j in range(numZero - i):
                sel[i, j] = 0
            for j in range(sel.shape[0] - numZero + i, sel.shape[0]):
                sel[i, j] = 0
        if (i - Radius + 1) > (Radius-1 - numZero):
            put = i - 2 * (Radius - 1) + numZero
            for j in range(put):
                sel[i, j] = 0
            for j in range(sel.shape[0] - put, sel.shape[0]):
                sel[i, j] = 0

    return sel


