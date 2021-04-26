import random
import cv2
import numpy as np
import scipy.ndimage as nd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def resizeImages(trainSeq, trainMasks, outputSize):
    """
    Function to resize images and their corresponding masks.
    :param trainSeq: Images for training or testing, of size (n, 512, 512) where n is the number of images
    :param trainMasks: Masks of size (n, 512, 512) where n is the number of images
    :param inputSize: A tuple containing the size to convert the images and masks to
    :return: outputSeq, outputMasks containing the new images and the corresponding masks
    """
    n, h, w = trainSeq.shape
    outputSeq = np.zeros((n, outputSize[0], outputSize[1]))
    outputMasks = np.zeros_like(outputSeq)

    for im in range(n):
        img = trainSeq[im, :, :]
        mask = trainMasks[im, :, :]
        outputSeq[im, :, :] = cv2.resize(img, outputSize)
        outputMasks[im, :, :] = cv2.resize(mask, outputSize)

    return outputSeq, outputMasks


def noiseAug(trainSeq, numImages):
    """
  Function to add random Gaussian Noise to given set of images.
  :param trainSeq: Images for training, of size (n, 512, 512) where n is the number of images
  :param numImages: Number of noisy augmented images required
  :return: output of size (n * numImages, 512, 512)
  """
    n, h, w = trainSeq.shape
    output = np.zeros((n * numImages, h, w))
    for im in range(n):
        img = trainSeq[im, :, :]
        for i in range(numImages):
            mean = random.uniform(0, 2)
            std = random.uniform(5, 30)
            noisyIm = img + np.random.normal(mean, std, img.shape)
            output[im * numImages + i, :, :] = noisyIm
    return output


def brightAug(trainSeq, numImages):
    """
  Function to alter brightness of an image.
  :param trainSeq: Images for training, of size (n, 512, 512) where n is the number of images
  :param numImages: Number of brightness augmented images required
  :return: output of size (n * numImages, 512, 512)
  """
    n, h, w = trainSeq.shape
    output = np.zeros((n * numImages, h, w))
    for im in range(n):
        img = trainSeq[im, :, :]
        img = img.astype(np.uint8)
        for i in range(numImages):
            alpha = np.random.uniform(low=0.3, high=1.8, size=1)
            out = cv2.addWeighted(src1=img, alpha=alpha, src2=0, beta=0, gamma=0)
            output[im * numImages + i, :, :] = out
    return output


def initNormResamp(trainSeq):
    """
    Function to clip, normalize and resample training images.
    :param trainSeq: Images for training, of size (n, 512, 512) where n is the number of images
    :return: output of size (n, 512, 512)
    """
    n, h, w = trainSeq.shape
    output = np.zeros_like(trainSeq)
    for i in range(n):
        img = trainSeq[i, :, :]
        clipped = np.clip(img, 0, 255)
        norm = cv2.normalize(clipped, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        zoomOut = nd.zoom(norm, 1, order=0)
        output[i, :, :] = zoomOut
    return output


def geomTransform(images, masks, numImages):
    """
    Function to perform random geometric transformations on a given set of images,
    and perform the same transforms on the ground truth mask of the image
    :param images: Images for training, of size (n, 512, 512) where n is the number of images
    :param masks: Corresponding masks for training of size (n, 512, 512)
    :param numImages: Number of transformed images required
    :return: returnImages and returnMasks of size (n*numImages, 512, 512)
    """
    dataGenArgs = dict(rotation_range=40,
                       width_shift_range=0.2,
                       height_shift_range=0.2,
                       rescale=1. / 255,
                       shear_range=0.2,
                       zoom_range=0.2,
                       horizontal_flip=True,
                       fill_mode='nearest')
    imageDataGen = ImageDataGenerator(**dataGenArgs)
    maskDataGen = ImageDataGenerator(**dataGenArgs)
    n, h, w = images.shape
    output = np.zeros((n * numImages, h, w))
    maskOut = np.zeros_like(output)
    for im in range(n):
        img = images[im, :, :]
        mask = masks[im, :, :]
        x = img.reshape((1,) + img.shape + (1,))
        y = mask.reshape((1,) + mask.shape + (1,))
        i = 0
        seed = 1
        for imageNew, maskNew in zip(imageDataGen.flow(x, batch_size=1, seed=seed),
                                     maskDataGen.flow(y, batch_size=1, seed=seed)):
            newim = imageNew[0]
            newm = maskNew[0]
            output[im * numImages + i, :, :] = newim[:, :, 0]
            maskOut[im * numImages + i, :, :] = newm[:, :, 0]
            i += 1
            if i >= numImages:
                break
    return output, maskOut


# Histogram Equalization
def histEq(trainSeq):
    """
    Function to perform Contrast Limited Adaptive Histogram Equalization on a given image.
    :param trainSeq: Images for training, of size (n, 512, 512) where n is the number of image
    :return: output of size (n, 512, 512)
    """
    n, h, w = trainSeq.shape
    output = np.zeros_like(trainSeq)
    for i in range(n):
        img = trainSeq[i, :, :].astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(img)
        output[i, :, :] = cl1
    return output


def genDataAugImages(trainImages, trainMasks):
    """
    Function to perform data augmentation - geometric transformations, brightness altering,
    noise injection and histogram equalization.
    :param trainImages: Images for training, of size (n, 512, 512) where n is the number of image
    :param trainMasks: Ground truth masks for training, of size (n, 512, 512) where n is the number of image
    :return: returnImages and returnMasks of augmented images, each of sizes (M, 512, 512), where M is the
             final image count
    """
    trainImages, trainMasks = resizeImages(trainImages, trainMasks, (256, 256))
    initialImages = initNormResamp(trainImages)
    geomImages, geomMasks = geomTransform(initialImages, trainMasks, 3)
    noisyImages = noiseAug(initialImages, 1)
    brightImages = brightAug(initialImages, 1)
    histImages = histEq(initialImages)

    print(initialImages.shape, geomImages.shape, noisyImages.shape, brightImages.shape, histImages.shape)

    returnImages = np.concatenate((initialImages, geomImages, noisyImages, brightImages, histImages), axis=0)
    returnMasks = np.concatenate(
        (trainMasks, geomMasks, np.tile(trainMasks, (1, 1, 1)), np.tile(trainMasks, (1, 1, 1)), trainMasks), axis=0)

#     returnImages = np.concatenate((initialImages, noisyImages, brightImages, histImages), axis=0)
#     returnMasks = np.concatenate(
#         (trainMasks, np.tile(trainMasks, (1, 1, 1)), np.tile(trainMasks, (1, 1, 1)), trainMasks), axis=0)

    shuffler = np.random.permutation(returnImages.shape[0])
    returnImages = returnImages[shuffler]
    returnMasks = returnMasks[shuffler]

    return returnImages, returnMasks
