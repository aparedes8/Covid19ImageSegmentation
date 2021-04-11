import tensorflow as tf
from tensorflow.keras.models import Model

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

    self.up6 = tf.keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.merge6 = tf.keras.layers.concatenate([self.drop4,self.up6], axis = 3)
    # merge6 = crop_and_concat()(drop4,up6)
    self.conv11 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv12 = tf.keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

    self.up7 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.merge7 = tf.keras.layers.concatenate([self.conv6,self.up7], axis = 3)
    # merge7 = crop_and_concat()(conv3,up7)
    self.conv13 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv14 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

    self.up8 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.merge8 = tf.keras.layers.concatenate([self.conv4,self.up8], axis = 3)
    # merge8 = crop_and_concat()(conv2,up8)
    self.conv15 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    self.conv16 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

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

    up6 = self.up6(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = self.merge6()
    conv11 = self.conv11(merge6)
    conv12 = self.conv12(conv11)

    up7 = self.up7(tf.keras.layers.UpSampling2D(size = (2,2))(conv12))
    merge7 = self.merge7()
    conv13 = self.conv11(merge6)
    conv14 = self.conv12(conv13)

    up8 = self.up7(tf.keras.layers.UpSampling2D(size = (2,2))(conv14))
    merge8 = self.merge8()
    conv15 = self.conv11(merge8)
    conv16 = self.conv12(conv15)

    up9 = self.up9(tf.keras.layers.UpSampling2D(size = (2,2))(conv16))
    merge9 = self.merge9()
    conv15 = self.conv11(merge8)
    conv16 = self.conv12(conv15)

    conv17 = self.conv17(conv16)
    conv18 = self.conv17(conv17)
    conv19 = self.conv17(conv18)

    return self.conv20(conv19)
