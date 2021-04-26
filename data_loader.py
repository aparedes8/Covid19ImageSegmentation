import tensorflow as tf
import numpy as np
import os
import nibabel as nib #Used to open nifTi or .nii files 
import cv2
from dataaug import resizeImages
###################################
# example from tensorflow
# from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds


@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (256, 256))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (256, 256))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask
def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (256, 256))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (256, 256))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

def example_ds():
  dataset , info = tfds.load('oxford_iiit_pet:3.*.*',with_info=True)
  TRAIN_LENGTH = info.splits['train'].num_examples
  BATCH_SIZE = 64
  BUFFER_SIZE = 1000
  STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
  # print(info)
  train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
  test = dataset['test'].map(load_image_test)

  train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
  train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
  test_dataset = test.batch(BATCH_SIZE)

  print("Dataset shape {} ".format(train_dataset))

  # train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
  # test = dataset['test'].map(load_image_test)
  print(train)

  return train_dataset,test_dataset,info
###################################

def load_data():
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
    print(train_imgs.shape)
    
    #normalize data between 0 and 1
    train_imgs = (train_imgs -  np.min(train_imgs))/(np.max(train_imgs)-np.min(train_imgs))
    
    #trian test split
    test_imgs = train_imgs[:,:,70:]
    train_imgs = train_imgs[:,:,:70]
    
    train_masks = train_masks_nib.get_fdata() 
    test_masks = train_masks[:,:,70:]
    train_masks = train_masks[:,:,:70]
    
    
    assert(train_imgs.shape == train_masks.shape)
    
    assert(train_imgs.shape[0:2] == test_imgs.shape[0:2])
    
    # Data formatched adding channel dimension and ensuring batch is at the front index 
    train_imgs = np.transpose(train_imgs)
    train_imgs = tf.cast(train_imgs[...,np.newaxis],tf.float32)
    
    train_masks = np.transpose(train_masks).astype(float)
    train_masks = train_masks[...,np.newaxis]

    test_imgs = np.transpose(test_imgs)
    test_imgs = tf.cast(test_imgs[...,np.newaxis],tf.float32)
    
    test_masks = np.transpose(test_masks).astype(float)
    
    train_imgs, train_masks = resizeImages(train_imgs, train_masks, (256, 256))
    test_imgs, test_masks = resizeImages(test_imgs, test_masks, (256, 256))

    print("Data loaded Train images shape : {} and Test images Shape {} ".format(train_imgs.shape,test_imgs.shape))
    print("Data loaded Train masks shape : {} and Test masks Shape {} ".format(train_imgs.shape,test_imgs.shape))

    return train_imgs,test_imgs,train_masks,test_masks
    


def convert_tf_dataset(train_imgs,test_imgs,train_masks,test_masks,BATCH_SIZE=2):
  #create tensorflow  tensor formated datasets from our numpy matrices
  train_ds = tf.data.Dataset.from_tensor_slices(
    (train_imgs, train_masks)).shuffle(train_imgs.shape[0]).batch(BATCH_SIZE)
  test_ds = tf.data.Dataset.from_tensor_slices((test_imgs, test_masks)).batch(BATCH_SIZE)
  for image,mask in train_ds.take(1):
    sample_image,sample_mask = image,mask

  print("DS img shape:{} mask shape {}".format(sample_image.shape,sample_mask.shape))

  return train_ds,test_ds