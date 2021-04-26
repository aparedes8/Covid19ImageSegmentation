import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from model import unet
from data_loader import *

import tensorflow_datasets as tfds

from IPython.display import clear_output
import matplotlib.pyplot as plt

# dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

# def normalize(input_image, input_mask):
#   input_image = tf.cast(input_image, tf.float32) / 255.0
#   input_mask -= 1
#   return input_image, input_mask

# @tf.function
# def load_image_train(datapoint):
#   input_image = tf.image.resize(datapoint['image'], (128, 128))
#   input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

#   if tf.random.uniform(()) > 0.5:
#     input_image = tf.image.flip_left_right(input_image)
#     input_mask = tf.image.flip_left_right(input_mask)

#   input_image, input_mask = normalize(input_image, input_mask)

#   return input_image, input_mask

# def load_image_test(datapoint):
#   input_image = tf.image.resize(datapoint['image'], (128, 128))
#   input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

#   input_image, input_mask = normalize(input_image, input_mask)

#   return input_image, input_mask


# TRAIN_LENGTH = info.splits['train'].num_examples

TRAIN_LENGTH = 70
BATCH_SIZE = 2
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_imgs,test_imgs,train_masks,test_masks = load_data()
train_dataset,test_dataset = convert_tf_dataset(train_imgs,test_imgs,train_masks,test_masks,BATCH_SIZE)

# print(info)
# print(dataset['image'].shape)

# train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
# test = dataset['test'].map(load_image_test)
# print(train)

# train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
# train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
# test_dataset = test.batch(BATCH_SIZE)
# print(train_dataset)

def display(display_list,num=0):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    if(display_list[i].ndim == 3):
      plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    else:
      plt.imshow(display_list[i])
    plt.axis('off')
  # plt.show()
  plt.savefig("./results/output _{}.png".format(num))


# for image, mask in train.take(1):
#   sample_image, sample_mask = image, mask
#   print(sample_image.shape)
#   print(sample_mask.shape)
# display([sample_image, sample_mask])

OUTPUT_CHANNELS = 4

base_model = tf.keras.applications.MobileNetV2(input_shape=[512, 512, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]


def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[512, 512, 1])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(OUTPUT_CHANNELS)
# model = unet((512,512,1),4)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()


tf.keras.utils.plot_model(model, show_shapes=True)


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    idx=0
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)],idx)
      idx+=1
  # else:
  #   display([sample_image, sample_mask,
  #            create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


EPOCHS = 4
VAL_SUBSPLITS = 5
# VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
VALIDATION_STEPS = 30/BATCH_SIZE/VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig("./results/test.png")

show_predictions(test_dataset, 3)