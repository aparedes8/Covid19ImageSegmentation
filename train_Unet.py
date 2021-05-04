###################################
# Library Imports
import tensorflow as tf
import numpy as np
import os
import nibabel as nib #Used to open nifTi or .nii files 
import matplotlib.pyplot as plt 
import cv2
from data_loader import *
from model import unet

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
x_train,x_test,y_train,y_test = load_data()
train_ds,test_ds = convert_tf_dataset(x_train,x_test,y_train,y_test,BATCH_SIZE=BATCH_SIZE)

###################################
'''
Lets display a sample image
'''
sample = x_train[0,:,:,0]

plt.imsave("sample_ct.png",sample)

###################################
'''
Build Unet model
'''
model = unet()
y = model(x_train[np.newaxis,0,...])


###################################
'''
Genrate optimizer and loss function and apply them to our unet model
'''
opt =tf.keras.optimizers.Adam(LR)
loss =tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.Accuracy()] 
model.compile(optimizer=opt,loss=loss,metrics=metrics)

###################################
'''
Train our model
'''
model.summary()

history = model.fit(train_ds,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=2,validation_data=test_ds)
print(history.history)

results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print("test loss, test acc:", results)

print(history.history.keys())

# summarize history for accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()