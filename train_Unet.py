###################################
# Library Imports
import tensorflow as tf
import numpy as np
import os
import nibabel as nib #Used to open nifTi or .nii files 
import matplotlib.pyplot as plt 

###################################
'''
Hyperparameters
'''
BATCH_SIZE = 2
EPOCHS = 5
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


# print(train_imgs.shape) # checking size of dataset images
# print(train_masks.shape) # checking size of  pixelwise mask labels
assert(train_imgs.shape == train_masks.shape)

test_imgs_filename = './data/test/val_im.nii.gz'
test_imgs_nib = nib.load(test_imgs_filename)

#convert nifti objects in to a numpy array
test_imgs = test_imgs_nib.get_fdata() 

# print(test_imgs.shape)
assert(train_imgs.shape[0:2] == test_imgs.shape[0:2])

# add channels dimension = 1 for greyscale images
# train_imgs = np.expand_dims(train_imgs,axis=2)
# test_imgs = np.expand_dims(test_imgs,axis=2)
channel_sized = np.expand_dims(train_imgs,axis=2)
INPUT_SHAPE = channel_sized.shape[0:3]
# print(INPUT_SHAPE)

###################################
'''
Lets display a sample image
'''
sample = train_imgs[:,:,0]

plt.imsave("sample_ct.png",sample)

###################################
'''
Build crop and concat layer regular tf.keras.concate has a odd shape mismatch
'''
class crop_and_concat(tf.keras.layers.Layer):
    def __init__(self):
        super(crop_and_concat, self).__init__()

    def call(self, x1,x2):
        with tf.name_scope("crop_and_concat"):
            x1_shape = tf.shape(x1)
            x2_shape = tf.shape(x2)
            # offsets for the top left corner of the crop
            offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
            size = [-1, x2_shape[1], x2_shape[2], -1]
            x1_crop = tf.slice(x1, offsets, size)
            return tf.concat([x1_crop, x2], 3)

###################################
'''
Build Unet model
'''
def unet(input_size = INPUT_SHAPE,batch_size=BATCH_SIZE):
    inputs = tf.keras.layers.Input(input_size,batch_size)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    print(drop4.shape)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
    print(up6.shape)
    merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
    # merge6 = crop_and_concat()(drop4,up6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
    # merge7 = crop_and_concat()(conv3,up7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
    # merge8 = crop_and_concat()(conv2,up8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
    merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
    # merge9 = crop_and_concat()(conv1,up9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = tf.keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model= tf.keras.Model(inputs,conv10)

    return model

model = unet()

###################################
'''
Genrate optimizer and loss function and apply them to our unet model
'''
opt =tf.keras.optimizers.Adam(LR)
loss =tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=opt,loss=loss)

###################################
'''
Train our model
'''
model.summary()

# train_imgs = np.squeeze(train_imgs,axis=0)
print(train_imgs.shape)
history = model.fit(x=channel_sized,y=train_masks,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=1)

