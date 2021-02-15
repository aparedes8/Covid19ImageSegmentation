###################################
# Library Imports
import tensorflow as tf
import numpy as np
import os
from nibabel.testing import data_path #Used to open nifTi or .nii files
import nibabel as nib
###################################
'''
Dataset loader obtained from the link below
https://medicalsegmentation.com/covid19/

# 100 total train images
# resolution = 512 x 512 
'''
train_imgs_filename = './data/train/tr_im.nii.gz'
train_masks_filename = './data/train/tr_mask.nii.gz'

train_imgs = nib.load(train_imgs_filename) 
train_masks = nib.load(train_masks_filename) 


print(train_imgs.shape) # checking size fo dataset images
print(train_masks.shape) # checking size of  pixelwise mask labels
assert(train_imgs.shape == train_masks.shape)
