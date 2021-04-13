import tensorflow as tf
from tensorflow.keras import layers,Model


def unet_model():
    # declaring the input layer
    # Input layer expects an RGB image, in the original paper the network consisted of only one channel.
    inputs = layers.Input(shape=(512, 512, 3))
    # first part of the U - contracting part
    c0 = layers.Conv2D(64, activation='relu', kernel_size=3)(inputs)
    c1 = layers.Conv2D(64, activation='relu', kernel_size=3)(c0)  # This layer for concatenating in the expansive part
    c2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c1)

    c3 = layers.Conv2D(128, activation='relu', kernel_size=3)(c2)
    c4 = layers.Conv2D(128, activation='relu', kernel_size=3)(c3)  # This layer for concatenating in the expansive part
    c5 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c4)

    c6 = layers.Conv2D(256, activation='relu', kernel_size=3)(c5)
    c7 = layers.Conv2D(256, activation='relu', kernel_size=3)(c6)  # This layer for concatenating in the expansive part
    c8 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c7)

    c9 = layers.Conv2D(512, activation='relu', kernel_size=3)(c8)
    c10 = layers.Conv2D(512, activation='relu', kernel_size=3)(c9)  # This layer for concatenating in the expansive part
    c11 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c10)

    c12 = layers.Conv2D(1024, activation='relu', kernel_size=3)(c11)
    c13 = layers.Conv2D(1024, activation='relu', kernel_size=3, padding='valid')(c12)

    # We will now start the second part of the U - expansive part
    t01 = layers.Conv2DTranspose(512, kernel_size=2, strides=(2, 2), activation='relu')(c13)
    crop01 = layers.Cropping2D(cropping=(4, 4))(c10)

    concat01 = layers.concatenate([t01, crop01], axis=-1)

    c14 = layers.Conv2D(512, activation='relu', kernel_size=3)(concat01)
    c15 = layers.Conv2D(512, activation='relu', kernel_size=3)(c14)

    t02 = layers.Conv2DTranspose(256, kernel_size=2, strides=(2, 2), activation='relu')(c15)
    crop02 = layers.Cropping2D(cropping=(16, 16))(c7)

    concat02 = layers.concatenate([t02, crop02], axis=-1)

    c16 = layers.Conv2D(256, activation='relu', kernel_size=3)(concat02)
    c17 = layers.Conv2D(256, activation='relu', kernel_size=3)(c16)

    t03 = layers.Conv2DTranspose(128, kernel_size=2, strides=(2, 2), activation='relu')(c17)
    crop03 = layers.Cropping2D(cropping=(40, 40))(c4)

    concat03 = layers.concatenate([t03, crop03], axis=-1)

    c18 = layers.Conv2D(128, activation='relu', kernel_size=3)(concat03)
    c19 = layers.Conv2D(128, activation='relu', kernel_size=3)(c18)

    t04 = layers.Conv2DTranspose(64, kernel_size=2, strides=(2, 2), activation='relu')(c19)
    crop04 = layers.Cropping2D(cropping=(88, 88))(c1)

    concat04 = layers.concatenate([t04, crop04], axis=-1)

    c20 = layers.Conv2D(64, activation='relu', kernel_size=3)(concat04)
    c21 = layers.Conv2D(64, activation='relu', kernel_size=3)(c20)

    # This is based on our dataset. The output channels are 3, think of it as each pixel will be classified
    # into three classes, but I have written 4 here, as I do padding with 0, so we end up have four classes.
    outputs = layers.Conv2D(4, kernel_size=1)(c21)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-netmodel")
    return model


class Unet(Model):
  def __init__(self):
    super(Unet, self).__init__()

    self.conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv2 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.conv3 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv4 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.conv5 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv6 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    self.conv7 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv8 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.drop4 = tf.keras.layers.Dropout(0.5)
    self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

    self.conv9 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv10 = tf.keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.drop5 = tf.keras.layers.Dropout(0.5)

    self.upsamp1 = tf.keras.layers.UpSampling2D(size = (2,2))
    self.up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.merge6 = tf.keras.layers.concatenate([self.drop4,self.up6], axis = 3)
    # merge6 = crop_and_concat()(drop4,up6)
    self.conv11 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv12 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

    self.upsamp2 = tf.keras.layers.UpSampling2D(size = (2,2))
    self.up7 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.merge7 = tf.keras.layers.concatenate([self.conv6,self.up7], axis = 3)
    # merge7 = crop_and_concat()(conv3,up7)
    self.conv13 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv14 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

    self.upsamp3 = tf.keras.layers.UpSampling2D(size = (2,2))
    self.up8 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.merge8 = tf.keras.layers.concatenate([self.conv4,self.up8], axis = 3)
    # merge8 = crop_and_concat()(conv2,up8)
    self.conv15 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv16 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

    self.upsamp4 = tf.keras.layers.UpSampling2D(size = (2,2))
    self.up9 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.merge9 = tf.keras.layers.concatenate([self.conv2,self.up9], axis = 3)
    # merge9 = crop_and_concat()(conv1,up9)
    self.conv17 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv18 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv19 = tf.keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv20 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')


  def call(self, x):
    conv1 = self.conv1(x)
    conv2 = self.conv2(conv1)
    pool1 = self.pool1(conv2)

    conv3 = self.conv3(pool1)
    conv4 = self.conv4(conv3)
    pool2 = self.pool2(conv4)

    conv5 = self.conv5(pool2)
    conv6 = self.conv6(conv5)
    pool3 = self.pool3(conv6)

    conv7 = self.conv7(pool3)
    conv8 = self.conv8(conv7)
    drop4 = self.drop4(conv8)
    pool4 = self.pool4(drop4)

    conv9 = self.conv9(pool4)
    conv10 = self.conv10(conv9)
    drop5 = self.drop5(conv10)

    upsamp1 = self.upsamp1(drop5)
    up6 = self.up6(upsamp1)
    merge6 = self.merge6()
    conv11 = self.conv11(merge6)
    conv12 = self.conv12(conv11)

    upsamp2 = self.upsamp2(conv12)
    up7 = self.up7(upsamp2)
    merge7 = self.merge7()
    conv13 = self.conv11(merge6)
    conv14 = self.conv12(conv13)

    upsamp3 = self.upsamp3(conv14)
    up8 = self.up7(upsamp3)
    merge8 = self.merge8()
    conv15 = self.conv11(merge8)
    conv16 = self.conv12(conv15)

    upsamp4 = self.upsamp4(conv16)
    up9 = self.up9(upsamp4)
    merge9 = self.merge9()
    conv15 = self.conv11(merge8)
    conv16 = self.conv12(conv15)

    conv17 = self.conv17(conv16)
    conv18 = self.conv17(conv17)
    conv19 = self.conv17(conv18)

    return self.conv20(conv19)
