
import tensorflow as tf
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load():

    ############################################################

    train_imgs_filename = './data/train/tr_im.nii.gz'
    train_masks_filename = './data/train/tr_mask.nii.gz'

    train_imgs_nib = nib.load(train_imgs_filename)
    train_masks_nib = nib.load(train_masks_filename)

    ############################################################

    train_imgs = train_imgs_nib.get_fdata()

    test_imgs = train_imgs[:, :, 70:].transpose(2,0,1).reshape(30, 512, 512, 1)
    train_imgs = train_imgs[:, :, :70].transpose(2,0,1).reshape(70, 512, 512, 1)

    mu, sigma = np.mean(train_imgs), np.std(train_imgs)
    train_imgs = (train_imgs - mu) / sigma
    test_imgs = (test_imgs - mu) / sigma

    ############################################################

    train_masks = train_masks_nib.get_fdata()
    test_masks = train_masks[:, :, 70:].transpose(2,0,1).reshape(30, 512, 512)
    train_masks = train_masks[:, :, :70].transpose(2,0,1).reshape(70, 512, 512)

    ############################################################

    return (train_imgs, train_masks), (test_imgs, test_masks)

