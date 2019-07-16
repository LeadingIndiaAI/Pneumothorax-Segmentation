
'''
Pneumothorax Detection using Mask_RCNN. Competiton hosted by 'Society for Imaging Informatics in Medicine (SIIM)' on https://www.kaggle.com
Competition Link: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation
Code credits: https://www.kaggle.com/soumikrakshit/pneumothorax-segmentation-using-unet-in-tensorflow

'''


import numpy as np
import pandas as pd
import cv2
from itertools import groupby
from imageio import imread
from random import randint
from tqdm import tqdm_notebook
from glob import glob
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.utils.vis_utils import model_to_dot
from keras import backend as K

TRAIN_SEED = randint(1, 1000)
VALIDATION_SEED = randint(1, 1000)


train_image_data_generator = ImageDataGenerator(
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rotation_range = 5,
    zoom_range = 0.1,
    rescale = 1.0 / 255.0
).flow_from_directory(
    ".../PTD1024/train_img",
    target_size = (1024, 1024),
    color_mode = 'grayscale',
    batch_size = 1,
    seed = TRAIN_SEED
)

train_mask_data_generator = ImageDataGenerator(
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rotation_range = 5,
    zoom_range = 0.1,
    rescale = 1.0 / 255.0
).flow_from_directory(
    ".../PTD1024/train_mask",
    target_size = (1024, 1024),
    color_mode = 'grayscale',
    batch_size = 1,
    seed = TRAIN_SEED
)

validation_image_data_generator = ImageDataGenerator(rescale = 1.0 / 255.0).flow_from_directory(
    ".../PTD1024/train_img",
    target_size = (1024, 1024),
    color_mode = 'grayscale',
    batch_size = 1,
    seed = VALIDATION_SEED,
)

validation_mask_data_generator = ImageDataGenerator(rescale = 1.0 / 255.0).flow_from_directory(
    ".../PTD1024/train_mask",
    target_size = (1024, 1024),
    color_mode = 'grayscale',
    batch_size = 1,
    seed = VALIDATION_SEED,
)


def mask_to_rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1
    return " " + " ".join(rle)



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def build_unet(shape):
    input_layer = Input(shape = shape)
    
    conv1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(input_layer)
    conv1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv4)
    pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv5)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides = (2, 2), padding = 'same')(conv5), conv4], axis = 3)
    conv6 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(up6)
    conv6 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same')(conv6), conv3], axis = 3)
    conv7 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(up7)
    conv7 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same')(conv7), conv2], axis = 3)
    conv8 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(up8)
    conv8 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same')(conv8), conv1], axis = 3)
    conv9 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(up9)
    conv9 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation = 'sigmoid')(conv9)
    
    return Model(input_layer, conv10)

model = build_unet((1024, 1024, 1))
model.summary()
model.compile(optimizer =Adam(lr=0.000000001), loss = dice_coef_loss, metrics = [dice_coef, 'binary_accuracy'])



weight_saver = ModelCheckpoint(
    'model_checkpoint.h5',
    monitor = 'val_dice_coeff',
    save_best_only = True,
    mode = 'min',
    save_weights_only = True
)

reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor = 'val_loss', factor = 0.5,
    patience = 3, verbose = 1,
    mode = 'min',
    cooldown = 2, min_lr = 1e-2
)

early = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 15
)

def train_data_generator(image_generator, mask_generator):
    while True:
        x_batch, _ = train_image_data_generator.next()
        y_batch, _ = train_mask_data_generator.next()
        yield x_batch, y_batch

def validation_data_generator(image_generator, mask_generator):
    while True:
        x_batch, _ = validation_image_data_generator.next()
        y_batch, _ = validation_mask_data_generator.next()
        yield x_batch, y_batch


history = model.fit_generator(
    train_data_generator(
        train_image_data_generator,
        train_mask_data_generator
    ),
    epochs = 200,
    steps_per_epoch = 670,
    validation_steps = 670,
    validation_data = validation_data_generator(
        validation_image_data_generator,
        validation_mask_data_generator
    ),
    verbose = 1,
    callbacks = [
        weight_saver,
        early,
        reduce_lr_on_plateau
    ]
)




rle, image_id = [], []
for file in tqdm_notebook(glob('.../PTD1024/test_img/test/*')):
    image = imread(file).reshape(1, 1024, 1024, 1)
    pred = model.predict(image).reshape(1024, 1024)
    image_id.append(file.split('/')[-1][:-4])
    encoding = mask_to_rle(pred, 1024, 1024)
    if encoding == ' ':
        rle.append('-1')
    else:
        rle.append(encoding)


submission = pd.DataFrame(data = {
    'ImageId' : image_id,
    'EncodedPixels' : rle
})
submission.head()


submission.to_csv('submission.csv', index = False)
model.save('model.h5')

