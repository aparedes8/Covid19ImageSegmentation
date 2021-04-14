###################################
# Library Imports
import tensorflow as tf
import numpy as np
import os
import nibabel as nib #Used to open nifTi or .nii files 
import matplotlib.pyplot as plt 
import cv2

###################################
'''
Hyperparameters
'''
BATCH_SIZE = 4
EPOCHS = 2
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

# train_imgs = (train_imgs * 255).astype(int)



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
# train_imgs = np.expand_dims(train_imgs,axis=3)


train_masks = np.transpose(train_masks).astype(float)
# train_masks = np.expand_dims(train_masks,axis=3)

print(train_masks.shape)
print(np.unique(train_masks))

print(train_imgs.shape)
print(np.unique(train_imgs))

print(train_masks.shape)
print(np.unique(train_masks))

print(train_imgs.shape)
print(np.unique(train_imgs))

###################################
'''
Lets display a sample image
'''
sample = train_imgs[0,:,:]

plt.imsave("sample_ct.png",sample)

###################################
'''
Build Unet model
'''
def unet(input_size = (512,512,1)):
    inputs = tf.keras.layers.Input(input_size)
    # print(inputs.shape)
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
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
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
    conv10 = tf.keras.layers.Conv2D(4, 1, activation = 'sigmoid')(conv9)
    final = tf.math.argmax(conv10,axis=3,output_type=tf.dtypes.int32)
    # final = tf.expand_dims(final,axis=3)

    model= tf.keras.Model(inputs,final)

    return model

model = unet()
y = model(train_imgs[np.newaxis,0,...])
print("fml")
# print(y.shape)
# print(y[0,0,0,0])

###################################
'''
Genrate optimizer and loss function and apply them to our unet model
'''
opt =tf.keras.optimizers.Adam(LR)
loss =tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
metrics = [tf.keras.metrics.Accuracy()] 
model.compile(optimizer=opt,loss=loss,metrics=metrics)

###################################
'''
Train our model
'''
model.summary()

history = model.fit(x=train_imgs,y=train_masks,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=2)
print(history.history)

results = model.evaluate(train_imgs, train_masks, batch_size=BATCH_SIZE)
print("test loss, test acc:", results)

x = np.linspace(1,len(history.history['loss'])+1,len(history.history['loss']))
print(x)
plt.title("Accuracy vs Loss Plot")
plt.plot(x,history.history['loss'],label='loss',color='k')
plt.plot(x,history.history['accuracy'],label='Accuracy',color='r')
plt.xlabel("EPOCHS")
plt.ylabel("Value")
plt.legend()
plt.savefig("./results/train_test_plot.png")