###################################
# Library Imports
import tensorflow as tf
import numpy as np
import os
import nibabel as nib #Used to open nifTi or .nii files 
import matplotlib.pyplot as plt 
from unet import Unet

###################################
'''
Hyperparameters
'''
BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-4

###################################
'''
Dataset loader obtained from the link below
https://medicalsegmentation.com/covid19/

# 100 total train images and 10 test images of
# resolution = 512 x 512 
'''
train_imgs_filename = './data/train/tr_im.nii.gz'
train_masks_filename = './data/train/tr_mask.nii.gz'

train_imgs_nib = nib.load(train_imgs_filename) 
train_masks_nib = nib.load(train_masks_filename)

#convert nifti objects in to a numpy array
train_imgs = train_imgs_nib.get_fdata() 
train_masks = train_masks_nib.get_fdata() 

assert(train_imgs.shape == train_masks.shape)

test_imgs_filename = './data/test/val_im.nii.gz'
test_imgs_nib = nib.load(test_imgs_filename)

#convert nifti objects in to a numpy array
test_imgs = test_imgs_nib.get_fdata() 

# print(train_imgs.shape)
assert(train_imgs.shape[0:2] == test_imgs.shape[0:2])


# Data formatched adding channel dimension and ensuring batch is at the front index 
train_imgs = np.transpose(train_imgs)
train_imgs = np.expand_dims(train_imgs,axis=3)


train_masks = np.transpose(train_masks)
train_masks = np.expand_dims(train_masks,axis=3)

###################################
'''
Lets display a sample image
'''
sample = train_imgs[0,:,:,0]

plt.imsave("sample_ct.png",sample)

###################################

print(train_masks.shape)