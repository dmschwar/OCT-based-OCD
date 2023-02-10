# visualizePreprocessing.py
# This script is used to help visualize the effects of the tomographyManager preprocessing functions.
# It works by loading the raw tomography data and passing it through the tm.dataSummoner() function to be preprocessed.
# It then saves the raw and preprocessed images side by side as a gif (WARNING: these gifs can be 50+ MB).

import imageio
import tensorflow as tf
# from tensorflow.image import encode_png

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

import TomographyManager
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input

import numpy as np
from skimage.transform import resize
from skimage import io
from skimage import filters
import sys
import os
import matplotlib
#control matplotlib backend
if sys.platform == 'darwin':
    matplotlib.use("tkAgg")
    print(matplotlib.get_backend())
    import matplotlib.pyplot as plt
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
# if 'linux' in sys.platform:
# tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

layersToFreeze = -1
postLayerSizes = [512]
dropoutRate = -1
imageInterval = [100, 150]

tm = TomographyManager.tomographyManager("/bigData/ovarianCancerDetection/machine_learning_OCT/", verbosity=True)

# get all animals that are 8 weeks old
allAnimalIDs = tm.getAnimalIDsByAge(8)

# get all indices (not just training indices)
allIndices = tm.getIndicesFromAnimalIDs(allAnimalIDs, imageInterval=imageInterval)

# get raw images
rawAnimalX = tm.getImagesFromIndices(allIndices[0:102])

# sampling an image to get the mean
sampleAnimalX, sampleAnimalY = tm.dataSummoner(allIndices[0:102], gaussianFilterSigma=5.0, preprocessingFilterDimensionality=2.0, modelType="vgg", numberOfChannels=3, stackedSlices=False)


imageset_1 = rawAnimalX
imageset_2 = sampleAnimalX.transpose(3, 0, 1, 2)[0]
imageset_2 = imageset_2 - np.full(imageset_2.shape, np.min(imageset_2))


stacked_images = np.append(imageset_1, imageset_2, axis=-1)
stacked_images = stacked_images.reshape((102, 752, 750*2))
stacked_images = [stacked_images[i, :, :] for i in range(stacked_images.shape[0])]
stacked_images = [resize(curImage, (376, 750)) for curImage in stacked_images]

imageio.mimsave('slices_after_processing.gif', stacked_images, fps=25)