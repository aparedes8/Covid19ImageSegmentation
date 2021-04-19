###################################
# Library Imports
import tensorflow as tf
import numpy as np
import os
import nibabel as nib #Used to open nifTi or .nii files 
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

# train_imgs = (train_imgs -  np.min(train_imgs))/(np.max(train_imgs)-np.min(train_imgs))

# train_imgs = (train_imgs * 255).astype(int)

train_masks = train_masks_nib.get_fdata() 

assert(train_imgs.shape == train_masks.shape)

print(train_masks.shape)
print(np.unique(train_masks))

print(train_imgs.shape)
print(np.unique(train_imgs))

###################################
'''
load saved unet model
'''
model = tf.keras.models.load_model("/content/drive/MyDrive/model/unet/")
model.summary()
###################################
'''
plot 10 train images their ground truth and the prediction from our unet model
'''
groundTruth = np.zeros((512,512,3))
prediction = np.zeros((512,512,3))

for i in range(0,6,3):
  img_idx = 0
  # plot grayscale ct scan
  plt.subplot(2,3,i+1)
  plt.imshow(train_imgs[:,:,img_idx],cmap='gray')
  # plot color labeled gt of ct scan
  plt.subplot(2,3,i+2)
  # get labels and color code them
  red_x,red_y = np.where(train_masks[:,:,img_idx] == 0)
  red_channels = np.zeros(len(red_x)).astype(int)

  green_x,green_y = np.where(train_masks[:,:,img_idx] == 1)
  green_channels = np.zeros(len(green_x)).astype(int)
  
  blue_x,blue_y = np.where(train_masks[:,:,img_idx] == 2)
  blue_channels = (np.ones(len(blue_x))*2).astype(int)

  yellow_x,yellow_y = np.where(train_masks[:,:,img_idx] == 2)
  yellow_channel_1 = (np.zeros(len(yellow_x))).astype(int)
  yellow_channel_2 = (np.ones(len(yellow_x))).astype(int)

  #set values
  groundTruth[red_x,red_y,red_channels] = 255
  groundTruth[green_x,green_y,green_channels] = 255
  groundTruth[blue_x,blue_y,blue_channels] = 255
  groundTruth[yellow_x,yellow_y,yellow_channel_1] = 255
  groundTruth[yellow_x,yellow_y,yellow_channel_2] = 255
  plt.imshow(groundTruth)
  #plot mdoel prediction
  plt.subplot(2,3,i+3)
  pred = model(train_imgs[np.newaxis,:,:,img_idx,np.newaxis])
  pred = np.argmax(pred,axis=3)
  pred = pred[0,:,:] #reduces unnnescary batch dimension ehre
  print(pred.shape)
  # get labels and color code them
  indices_red = np.where(pred == 0)
  indices_green = np.where(pred == 1)
  indices_blue = np.where(pred == 2)
  indices_yellow = np.where(pred == 3)
  #set values
  prediction[indices_red,0] = 255
  prediction[indices_green,1] = 255
  prediction[indices_blue,2] = 255
  prediction[indices_yellow,0] = 255
  prediction[indices_yellow,1] = 255
  plt.imshow(prediction)

plt.savefig("./results/training_visualization.png")



