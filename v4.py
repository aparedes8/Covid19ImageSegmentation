import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from data import load_data, tf_dataset
from model import build_model

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
        return tf.numpy_function(f, [y_true, y_pred], tf.float32)
    
if __name__ == "__main__":
    ###################################
    '''
    Hyperparameters
    '''
    BATCH_SIZE = 4
    EPOCHS = 3
    LR = 1e-7
    
    
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
    
    
    train_imgs = (train_imgs -  np.min(train_imgs))/(np.max(train_imgs)-np.min(train_imgs))
    
    train_imgs = (train_imgs * 255).astype(int)
    
    
    
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
    
    print(train_masks.shape)
    print(np.unique(train_masks))
    
    print(train_imgs.shape)
    print(np.unique(train_imgs))
    
    ###################################
    '''
    Lets display a sample image
    '''
    sample = train_imgs[0,:,:,0]
    
    plt.imsave("sample_ct.png",sample)
    
    ###################################
    
    model = build_model()

    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

    callbacks = [
        ModelCheckpoint("files/model.h5"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]

    train_steps = len(train_x)//batch
    valid_steps = len(valid_x)//batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    model.fit(train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks)