###################################
# Library Imports
import tensorflow as tf
import numpy as np
import os
from nibabel.testing import data_path #Used to open nifTi or .nii files
import nibabel as nib
import matplotlib.pyplot as plt 

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


print(train_imgs.shape) # checking size of dataset images
print(train_masks.shape) # checking size of  pixelwise mask labels
assert(train_imgs.shape == train_masks.shape)

test_imgs_filename = './data/test/val_im.nii.gz'
test_imgs_nib = nib.load(test_imgs_filename)

#convert nifti objects in to a numpy array
test_imgs = test_imgs_nib.get_fdata() 

print(test_imgs.shape)
assert(train_imgs.shape[0:2] == test_imgs.shape[0:2])

###################################
'''
Lets display a sample image
'''
sample = train_imgs[:,:,0]

plt.imsave("sample_ct.png",sample)

###################################
