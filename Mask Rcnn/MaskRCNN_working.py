
'''
Pneumothorax Detection using Mask_RCNN. Competiton hosted by 'Society for Imaging Informatics in Medicine (SIIM)' on https://www.kaggle.com
Competition Link: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation
Code credits: https://www.kaggle.com/hmendonca/mask-rcnn-and-medical-transfer-learning-siim-acr

'''




import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm, tqdm_notebook
import pandas as pd 
import glob

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)
#directory containing dataset
DATA_DIR = '.../PNToriginal/'

#directory to save models and logs
ROOT_DIR = '.../PNToriginal/worksix'

#directory of Mask RCNN, download it from: https://github.com/matterport/Mask_RCNN
new_DIR='.../PNToriginal/tmp/'

sys.path.append(os.path.join(new_DIR, 'Mask_RCNN'))
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

train_dicom_dir = os.path.join(DATA_DIR, 'dicom-images-train')
test_dicom_dir = os.path.join(DATA_DIR, 'dicom-images-test')

# get model with best validation score: https://www.kaggle.com/hmendonca/mask-rcnn-and-coco-transfer-learning-lb-0-155/
WEIGHTS_PATH = ".../PNToriginal/mask.rcnn.weights/mask_rcnn_coco.h5"


debug = False
IMAGE_DIM = 512

class DetectorConfig(Config):    
  
    NAME = 'Pneumothorax'
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 11
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  
    
    IMAGE_MIN_DIM = IMAGE_DIM
    IMAGE_MAX_DIM = IMAGE_DIM
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 12
    DETECTION_MAX_INSTANCES = 4
    DETECTION_MIN_CONFIDENCE = 0.90
    DETECTION_NMS_THRESHOLD = 0.1

    STEPS_PER_EPOCH = 350
    VALIDATION_STEPS = 120
    
    
    LOSS_WEIGHTS = {
        "rpn_class_loss": 10.0,
        "rpn_bbox_loss": 0.6,
        "mrcnn_class_loss": 6.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 2.4
    }

config = DetectorConfig()
config.display()

import os
import numpy as np 
import pandas as pd 
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from skimage.segmentation import mark_boundaries
from skimage.util import montage
from skimage.morphology import binary_opening, disk, label
import gc; gc.enable()

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

def multi_rle_encode(img, **kwargs):
    
    labels = label(img)
    if img.ndim > 2:
        return [rle2mask(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle2mask(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

def masks_as_image(rle_list, shape):
    all_masks = np.zeros(shape, dtype=np.uint8)
    for mask in rle_list:
        if isinstance(mask, str) and mask != '-1':
            all_masks |= rle2mask(mask, shape[0], shape[1]).T.astype(bool)
    return all_masks

from PIL import Image
from sklearn.model_selection import train_test_split

train_glob = '.../PNToriginal/dicom-images-train//*/*/*.dcm'
test_glob = '.../PNToriginal/dicom-images-test/*/*/*.dcm'

exclude_list = []
train_names = [f for f in sorted(glob.glob(train_glob)) if f not in exclude_list]
test_names = [f for f in sorted(glob.glob(test_glob)) if f not in exclude_list]

print(len(train_names), len(test_names))
print(train_names[0], test_names[0])
os.path.join(train_dicom_dir, train_names[0])

#training dataset
SEGMENTATION ='.../PNToriginal/train-rle.csv'
anns = pd.read_csv(SEGMENTATION)
anns.head()
anns.columns = ['ImageId', 'EncodedPixels']

#Postive samples
pneumothorax_anns = anns[anns.EncodedPixels != ' -1'].ImageId.unique().tolist()
print(f'Positive samples: {len(pneumothorax_anns)}/{len(anns.ImageId.unique())} {100*len(pneumothorax_anns)/len(anns.ImageId.unique()):.2f}%')


pneumothorax_fps_train = [fp for fp in train_names if fp.split('/')[-1][:-4] in pneumothorax_anns]



image_fps_train, image_fps_val = train_test_split(pneumothorax_fps_train, test_size=0.1, random_state=42)

test_image_fps = test_names

if debug:
    print('DEBUG subsampling from:', len(image_fps_train), len(image_fps_val), len(test_image_fps))
    image_fps_train = image_fps_train
    image_fps_val = image_fps_val

    
class DetectorDataset(utils.Dataset):
    """Dataset class for training our dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        
        self.add_class('pneumothorax', 1, 'Pneumothorax')
        
       
        for i, fp in enumerate(image_fps):
            image_id = fp.split('/')[-1][:-4]
            annotations = image_annotations.query(f"ImageId=='{image_id}'")['EncodedPixels']
            self.add_image('pneumothorax', image_id=i, path=fp, 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)
            
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If in grayscale, Convert to RGB for consistency
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
#         print(image_id, annotations)
        count = len(annotations)
        if count == 0 or (count == 1 and annotations.values[0] == ' -1'): #no_mask
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                mask[:, :, i] = rle2mask(a, info['orig_height'], info['orig_width']).T
                class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)

image_fps, image_annotations = train_names, anns

ds = pydicom.read_file(image_fps[0]) #reading the dicom images
image = ds.pixel_array 



ORIG_SIZE = 1024


#training_dataset
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()


#validation_dataset
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()


#Image_augmentation
augmentation = iaa.Sequential([
    iaa.OneOf([ #geometric transformation
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.04)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.001, 0.025)),
    ]),
    iaa.OneOf([ #brightness/contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ #blur/sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.2)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])


#get pixel statistics
images = []
for image_id in dataset_val.image_ids:
    image = dataset_val.load_image(image_id)
    images.append(image)

images = np.array(images)
config.MEAN_PIXEL = images.mean(axis=(0,1,2)).tolist()
VAR_PIXEL = images.var()
print(config.MEAN_PIXEL, VAR_PIXEL)


model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)


model.load_weights(WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

LEARNING_RATE = 0.0006


model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE*2,
            epochs=30,
            layers='heads',
            augmentation=None)  #no augmentation

history = model.keras_model.history.history

model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE,
            epochs=100 if debug else 100,
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]

model.train(dataset_train, dataset_val,
            learning_rate=LEARNING_RATE/2,
            epochs=100 if debug else 100,
            layers='all',
            augmentation=augmentation)

new_history = model.keras_model.history.history
for k in new_history: history[k] = history[k] + new_history[k]


epochs = range(1, len(history['loss'])+1)
pd.DataFrame(history, index=epochs)

best_epoch = np.argmin(history["val_loss"])
score = history["val_loss"][best_epoch]
print(f'Best Epoch:{best_epoch+1} val_loss:{score}')

# select trained model 
dir_names = next(os.walk(model.model_dir))[1]
key = config.NAME.lower()
dir_names = filter(lambda f: f.startswith(key), dir_names)
dir_names = sorted(dir_names)
print(dir_names)

if not dir_names:
    import errno
    raise FileNotFoundError(
        errno.ENOENT,
        "Could not find model directory under {}".format(self.model_dir))

fps = []
#go to last directory
for d in dir_names: 
    print(d)
    dir_name = os.path.join(model.model_dir, d)
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if checkpoints:
        
    	if best_epoch < len(checkpoints):
        	checkpoint = checkpoints[best_epoch]
    	else:
        	checkpoint = checkpoints[-1]
    	fps.append(os.path.join(dir_name, checkpoint))
    else: 
      print(f'No weight files in {dir_name}')

model_path = sorted(fps)[-1]
print('Found model {}'.format(model_path))

class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)


assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


sub = pd.read_csv('.../PNToriginal/sample_submission.csv')


positives = sub.groupby('ImageId').ImageId.count().reset_index(name='N').set_index('ImageId')
positives = positives.loc[positives.N > 1] positives.head()

#predicting and submission file
def predict(image_fps, filepath='submission.csv', min_conf=config.DETECTION_MIN_CONFIDENCE):
    
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    with open(filepath, 'w') as file:
        file.write("ImageId,EncodedPixels\n")

        for fp in tqdm_notebook(image_fps):
            image_id = fp.split('/')[-1][:-4]
            maks_written = 0
            
            if image_id in positives.index:
                ds = pydicom.read_file(fp)
                image = ds.pixel_array
                
                if len(image.shape) != 3 or image.shape[2] != 3:
                    image = np.stack((image,) * 3, -1)
                image, window, scale, padding, crop = utils.resize_image(
                    image,
                    min_dim=config.IMAGE_MIN_DIM,
                    min_scale=config.IMAGE_MIN_SCALE,
                    max_dim=config.IMAGE_MAX_DIM,
                    mode=config.IMAGE_RESIZE_MODE)

                results = model.detect([image])
                r = results[0]


                n_positives = positives.loc[image_id].N
                num_instances = min(len(r['rois']), n_positives)

                for i in range(num_instances):
                    if r['scores'][i] > min_conf and np.sum(r['masks'][...,i]) > 1:
                        mask = r['masks'][...,i].T*255

                        mask, _,_,_,_ = utils.resize_image(
                            np.stack((mask,) * 3, -1), 
                            min_dim=ORIG_SIZE,
                            min_scale=config.IMAGE_MIN_SCALE,
                            max_dim=ORIG_SIZE,
                            mode=config.IMAGE_RESIZE_MODE)
                        mask = (mask[...,0] > 0)*255

                        file.write(image_id + "," + mask2rle(mask, ORIG_SIZE, ORIG_SIZE) + "\n")
                        maks_written += 1
                    
               
                for i in range(n_positives - maks_written):
                    padding = 88750
                    file.write(image_id + f",{padding} {ORIG_SIZE*ORIG_SIZE - padding*2}\n")
                    maks_written += 1



            if maks_written == 0:
                file.write(image_id + ",-1\n")  

submission_fp = os.path.join(ROOT_DIR, 'submissionnew.csv')
predict(test_image_fps, filepath=submission_fp)
print(submission_fp)

sub = pd.read_csv(submission_fp)
print((sub.EncodedPixels != '-1').sum(), sub.ImageId.size, sub.ImageId.nunique())
print(sub.EncodedPixels.nunique(), (sub.EncodedPixels != '-1').sum()/sub.ImageId.nunique())

print('Unique samples:\n', sub.EncodedPixels.drop_duplicates()[:6])
sub.head(10)
