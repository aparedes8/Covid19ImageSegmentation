import tensorflow as tf
import tensorflow_datasets as tfds
import unet
# from plotImage import display
import nibabel as nib #Used to open nifTi or .nii files 

# Hyperparameters
BATCH_SIZE = 8
BUFFER_SIZE = 1000
EPOCHS = 20

# First step is to load the dataset
# Disable the progress bar.

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
# train_imgs = np.transpose(train_imgs)
# train_imgs = np.expand_dims(train_imgs,axis=3)


# train_masks = np.transpose(train_masks)
# train_masks = np.expand_dims(train_masks,axis=3)

###################################
#create tensorflow  tensor formated datasets from our numpy matrices
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_imgs, train_masks)).shuffle(train_imgs.shape[0]).batch(BATCH_SIZE)

###################################


# Shuffle, batch, cache and prefetch the dataset.
train_dataset = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# test_dataset = test.batch(BATCH_SIZE)


# # Check out how does the training set look like.
# for image, mask in train.take(1):
#   sample_image, sample_mask = image, mask
#   display([sample_image, sample_mask])

# Create the model.
model = unet.unet_model()
# Build the model with the input shape
# Image is RGB, so here the input channel is 3.
model.build(input_shape=(None,3,512, 512))
model.summary()

# Write model saving callback.
model_save_callback = tf.keras.callbacks.ModelCheckpoint(
    './model_checkpoint', monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch')

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_history = model.fit(train_dataset, epochs=EPOCHS, callbacks=[model_save_callback])