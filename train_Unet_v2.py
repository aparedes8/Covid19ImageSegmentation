###################################
# Library Imports
import tensorflow as tf
import numpy as np
import os
import nibabel as nib #Used to open nifTi or .nii files 
import matplotlib.pyplot as plt 
from unet import Unet

###################################
'''
Hyperparameters
'''
BATCH_SIZE = 2
EPOCHS = 30
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

###################################
'''
Lets display a sample image
'''
sample = train_imgs[0,:,:,0]

plt.imsave("sample_ct.png",sample)

###################################

print(train_masks.shape)

###################################
#create tensorflow  tensor formated datasets from our numpy matrices
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_imgs, train_masks)).shuffle(train_imgs.shape[0]).batch(BATCH_SIZE)

# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

###################################
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(LR)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
###################################
'''
Generate gradient caclulations and weight update steps from trian and test
'''
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)



###################################
'''
Train and test our model
'''
train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []


# model.summary()
train_len = len(list(train_ds))
# test_len = len(list(test_ds))
for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  cnt = 0
  for images, labels in train_ds:
    print('Training batch {} out of {}'.format(cnt, train_len))
    cnt += 1
    train_step(images, labels)
  cnt = 0
#   for test_images, test_labels in test_ds:
#     print('Testing batch {} out of {}'.format(cnt, test_len))
#     cnt += 1
#     test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )
  train_acc_list.append(train_accuracy.result())
  train_loss_list.append(train_loss.result())

  test_acc_list.append(test_accuracy.result())
  test_loss_list.append(test_loss.result())


epoch_list = np.arange(1,EPOCHS+1,1)
print(len(epoch_list))
print(len(train_acc_list))

plt.title("Accuracy vs Epoch Plot")
plt.plot(epoch_list,train_acc_list,label='Train Accuracy',color='r')
# plt.plot(epoch_list,test_acc_list,label='Test Accuracy',color='g')

plt.xlabel("EPOCHS")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("./results/Accuracy_plot.png")

plt.figure()
plt.title("Loss vs Epoch Plot")
# plt.plot(epoch_list,test_loss_list,label='Test loss',color='r')
plt.plot(epoch_list,train_loss_list,label='Train loss',color='g')
plt.xlabel("EPOCHS")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./results/loss_plot.png")