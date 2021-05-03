import tensorflow as tf
import numpy as np
import os
import nibabel as nib  # Used to open nifTi or .nii files
import cv2
from dataaug import resizeImages

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

    # convert nifti objects in to a numpy array
    train_imgs = train_imgs_nib.get_fdata()
    print(train_imgs.shape)

    # normalize data between 0 and 1
    train_imgs = (train_imgs - np.min(train_imgs)) / (np.max(train_imgs) - np.min(train_imgs))

    # trian test split
    test_imgs = train_imgs[:, :, 70:]
    train_imgs = train_imgs[:, :, :70]

    train_masks = train_masks_nib.get_fdata()
    test_masks = train_masks[:, :, 70:]
    train_masks = train_masks[:, :, :70]

    assert (train_imgs.shape == train_masks.shape)

    assert (train_imgs.shape[0:2] == test_imgs.shape[0:2])

    # Data formatched adding channel dimension and ensuring batch is at the front index
    train_imgs = np.transpose(train_imgs)
    train_imgs = tf.cast(train_imgs[..., np.newaxis], tf.float32)

    train_masks = np.transpose(train_masks).astype(float)
    train_masks = train_masks[..., np.newaxis]

    test_imgs = np.transpose(test_imgs)
    test_imgs = tf.cast(test_imgs[..., np.newaxis], tf.float32)

    test_masks = np.transpose(test_masks).astype(float)
    test_masks = test_masks[:, :, :, np.newaxis]

    train_imgs = tf.image.resize(train_imgs, [256, 256])
    train_masks = tf.image.resize(train_masks, [256, 256])
    test_imgs = tf.image.resize(test_imgs, [256, 256])
    test_masks = tf.image.resize(test_masks, [256, 256])

    test_masks = test_masks[:, :, :, 0]
    print(train_masks.shape, test_masks.shape)

    print("Data loaded Train images shape : {} and Test images Shape {} ".format(train_imgs.shape, test_imgs.shape))
    print("Data loaded Train masks shape : {} and Test masks Shape {} ".format(train_imgs.shape, test_imgs.shape))

    return train_imgs, test_imgs, train_masks, test_masks


def convert_tf_dataset(train_imgs, test_imgs, train_masks, test_masks, BATCH_SIZE=2):
    # create tensorflow  tensor formated datasets from our numpy matrices
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_imgs, train_masks)).shuffle(train_imgs.shape[0]).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((test_imgs, test_masks)).batch(BATCH_SIZE)
    for image, mask in train_ds.take(1):
        sample_image, sample_mask = image, mask

    print("DS img shape:{} mask shape {}".format(sample_image.shape, sample_mask.shape))

    return train_ds, test_ds