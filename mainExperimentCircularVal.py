########################################################################################################################
#Author: David Schwartz
# In the leave-3-out experiment, we'll iterate over every choice of a single holdout animal
# and hold out two random animals, (guaranteeing one positive and one negative) from the training for validation
#
# This script instantiates and tests the vgg on the available tomography data
# example arguments: -v -p /home/dmschwar/OCD/data/machine_learning_OCT -ii 200 400 -gSigma 10 -ds 6500 -cvf 3 -e 2

# -v -p a:bigData/machine_learning_OCT -op together -as 8 -ii 100 102 -gSigma 2 -bs 12 -kb 4 -e 25 -tp 25 -mt lstm -prs 2 -prt cnn -ps 2 -lstms 2 -ra 1.1 -dr 0.5 -pi /temp/

# -v -p /bigData/ovarianCancerDetection/machine_learning_OCT -op together -as 8 -ii 100 149 -gSigma 2 -bs 400 -kb 4 -e 25 -tp 25 -mt lstm -prs 128 32 -prt cnn -ps 32 64 -lstms 128 -ra 1.1 -dr 0.5 -pi /temp/
# -v -p /extra/dmschwar/bigData/machine_learning_OCT -op together -as 8 -ii 100 150 -gSigma 2 -bs 400 -e 100 -tp 25 -mt lstm -prs 2 2 -prt cnn -pot cnn -ps 32 16 -lstms 64 32 -ra 0.314159 -pi /extra/dmschwar/intermediateData/OCD/
# -v -p /groups/ditzler/dmschwar/bigData/machine_learning_OCT -op together -as 8 -ii 100 150 -gSigma 4 -bs 400 -e 100 -tp 25 -mt vgg -lf 2 -ps 512 -ra 1.314159 -dr 0.5 -pi /groups/ditzler/dmschwar/intermediateData/OCD/
#
########################################################################################################################
import subprocess as sp
import copy
import sklearn
from sklearn import preprocessing
import random
import argparse
import os
import sys
from sys import platform
import TomographyManager
import numpy as np
import skimage
from skimage import filters
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut
from sklearn import preprocessing
import tensorflow
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras import losses
from tensorflow.python.client import device_lib
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import utils, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Conv2D, ConvLSTM2D, Conv3D, \
    MaxPool2D, LeakyReLU, AveragePooling1D,LSTM, Bidirectional, TimeDistributed,BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint
# if tf.__version__ == "2.4.0":
#     from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
# else:
# from tensorflow.keras.utils import multi_gpu_model
import imageio
import matplotlib
import shelve
import time
import scipy
import itertools
import pylab
# from pylab import rcParams
# rcParams['figure.figsize'] = 6.5, 2.5

#set random seeds once
os.environ['PYTHONHASHSEED']=str(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(a=42, version=2)
tf.random.set_seed(42)



#control matplotlib backend
if sys.platform == 'darwin':
    matplotlib.use("tkAgg")
    print(matplotlib.get_backend())
    import matplotlib.pyplot as plt
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
if 'linux' in sys.platform:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pylab as plt

gpuNameList = None
#if we have gpus available, enumerate them
# if tensorflow.test.gpu_device_name():
#     print('using gpu configuration')
#     config = tensorflow.ConfigProto()
#     config.gpu_options.allow_growth = True
#
#     #count the number of gpus and generate a list of device strings on which we can dispatch our folds
#     localDevicePrototypes = device_lib.list_local_devices()
#     gpuNameList = [curProto.name for curProto in localDevicePrototypes]

########################################################################################################################
#method to plot a confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    print(cm.shape)
    # Only use the labels that appear in the data
    # classes = classes[sklearn.utils.multiclass.unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    if (cm.shape[0] > 1): 
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()  
    return fig, ax

########################################################################################################################
#define default parameters
defaultVerbosity = True
defaultGraphical = True
defaultSigma = 0
defaultPath = os.getcwd()
defaultImageInterval = [50, 150]
defaultAgeSelection = 0
defaultReluAlpha = None

#set up argument parser
parser = argparse.ArgumentParser()

#define the arguments
parser.add_argument('-v', '--verbose', help='increased console verbosity',
                    action='store_true', required=False)
parser.add_argument('-g', '--graphical', help='plot graphics as we run',
                    action='store_true', required=False)
parser.add_argument('-p', '--pathToTomographyFiles', help='absolute path to the machine_learning_OCT directory',
                    type=str, required=True)
parser.add_argument('-pi', '--pathToIntermediateData', help='absolute path to location to store .h5 and .shelf files',
                    type=str, required=True)
parser.add_argument('-ii', '--imageInterval', help='interval of slices that are to be included in the dataset',
                    type=int, nargs='+', required=False)
parser.add_argument('-gSigma', '--gaussianFilterSigma', help='width of the gaussian filter to apply we will use to smooth the images',
                    type=float, required=False)
parser.add_argument('-pfd', '--preprocessingFilterDimensionality', help='dimensionality of the gaussian filter',
                    type=float, required=False)
parser.add_argument('-lf', '--layersToFreeze', help='number of layers of the pretrained model to NOT fine tune',
                    type=int, required=False)
parser.add_argument('-e', '--fineTuningEpochs',
                    help='number of epochs over which to fine tune the model',
                    type=int, required=False)
parser.add_argument('-ed', '--edgeDetection',
                    help='controls if we pass signals through canny edge detection in preprocessing if float and positive',
                    type=float, required=False)
parser.add_argument('-std', '--standardize',
                    help='controls if we standardize data in preprocessing instead of simply centering and scaling to (-1,1)',
                    type=bool, required=False)
parser.add_argument('-bs', '--batchSize',
                    help='number of images to read into memory for current batch in training and testing',
                    type=int, required=False)
parser.add_argument('-lt', '--lossThreshold', help='threshold on loss below which training stops',
                    type=float, required=False)
parser.add_argument('-dr', '--dropoutRate', help='dropped fraction (ie  1 - keep_keep_rate in dropout)',
                    type=float, required=False)
parser.add_argument('-ps', '--postLayerSizes', help='controls sizes of the post-vgg layers',
                    type=int, nargs='+', required=False)
parser.add_argument('-prs', '--preLayerSizes', help='controls sizes of the pre-lstm layers (right now, this is only implemented with the lstm)',
                    type=int, nargs='+', required=False)
parser.add_argument('-prt', '--preLayerType', help='controls if we have any processing between the input layer and the LSTM layers',
                    type=str, required=False)
parser.add_argument('-pot', '--postLayerType', help='controls if we have any processing between the LSTM layers and the output layers',
                    type=str, required=False)
parser.add_argument('-lstms', '--lstmLayerSizes', help='controls sizes of the lstm layers',
                    type=int, nargs='+', required=False)
parser.add_argument('-c3ds', '--c3dLayerSizes', help='controls sizes of the 3d convolutional layers',
                    type=int, nargs='+', required=False)
parser.add_argument('-pst', '--positiveSlicesThreshold', help='number of slices per animal that must be classified as '
                                                              'in order to declare that the animal is positive ',
                    type=int, required=False)
parser.add_argument('-kb', '--kBatchSize',
                    help='number of images to read into memory for current batch in training and testing',
                    type=int, required=False)
parser.add_argument('-ra', '--reluAlpha',
                    help='controls the value of parameter alpha to the leaky relu activation in the re-encoding layer',
                    type=float, required=False)
parser.add_argument('-rs', '--replications',
                    help='controls the number of replications of the main experiment run in determining an interpolated region of typical ROCs',
                    type=int, required=False, default=1)
parser.add_argument('-as', '--ageSelection',
                    help='age of animal to select:: -1: all; 4MO: 4 months old; 8MO: 8 months old',
                    type=int, required=False)
parser.add_argument('-mt', '--modelType', help='model type: select vgg, mlp, or xception',
                    type=str, required=True)
parser.add_argument('-mds', '--modelDescriptorString', help='if supplied, then this determines the mds variable (which is used to identify a location on the hard drive to write results and intermediate data)',
                    type=str, required=False, default=None)
parser.add_argument('-tp', '--trainingPatience', help='patience for early stopping (i.e. number of epochs to wait before halting training when delta(loss) < lossThreshold)',
                    type=int, required=False)
parser.add_argument('-tv', '--trainOnValidationData', help='controls if we train on the validation data at the end of the main training session',
                    type=bool, required=False)
parser.add_argument('-op', '--organParadigm', help='controls if the experiment tests both of the test animal organs together or as separate subjects.'
                                                   'Usage: "together", "separate"',
                    type=str, required=False)
parser.add_argument('-ss', '--stackedSlices', help='if True, 3 consecutive tomography slices are stacked to fill the '
                                                   'RGB channels, if False a single slice is copied 3 times',
                    action='store_true', required=False)
parser.add_argument('-plr', '--plotLearning',
                    help='controls if we plot loss and accuracy side by side as training proceeds',
                    type=bool, required=False)


# parser.add_argument('-np', '--neymanPearson', help='if present, use a neyman pearson hypothesis test with '
#                                                    'probability given by the argument to classify',
#                     type=float, required=False)

# parser.add_argument('-ds', '--decisionStrategy', help='controls which decision strategy we use: absence or 0 selects a \
#                                                     uniformly weighted voting procedure across all samples for the given animal; \
#                                                     1 selects the most sensitive strategy (any positive slice is evidence \
#                                                     for positive classification',\
#                     type=int, nargs='+', required=False)

# parser.add_argument('-cvf', '--numberOfCrossValidationFolds',
#                     help='number of folds to use in the cross validation experiment',
#                     type=int, required=False)

# parser.add_argument('k', '--kToLeaveOut', help='number of animals to exclude from the training set',
#                     type=int, required=False)

#parse the arguments
arguments = parser.parse_args()
verbosity = arguments.verbose if arguments.verbose is not None else False #default verbosity set to false
if (verbosity):
    print(arguments)
usingGraphics = arguments.graphical if arguments.graphical is not None else defaultGraphical
pathToTomographyFiles = arguments.pathToTomographyFiles if arguments.pathToTomographyFiles is not None else defaultPath#default path is root directory
pathToIntermediateData = arguments.pathToIntermediateData if arguments.pathToIntermediateData is not None else defaultPath
imageInterval = arguments.imageInterval if arguments.imageInterval is not None and len(arguments.imageInterval) == 2 else None
gaussianFilterSigma = arguments.gaussianFilterSigma if arguments.gaussianFilterSigma is not None else defaultSigma
preprocessingFilterDimensionality = arguments.preprocessingFilterDimensionality if arguments.preprocessingFilterDimensionality is not None else 2
# datasetHalfSize = arguments.datasetHalfSize if arguments.datasetHalfSize is not None else -1
layersToFreeze = arguments.layersToFreeze if arguments.layersToFreeze is not None else -1
# numberOfCrossValidationFolds = arguments.numberOfCrossValidationFolds \
#     if arguments.numberOfCrossValidationFolds is not None else 2
fineTuningEpochs = arguments.fineTuningEpochs if arguments.fineTuningEpochs is not None else 1
batchSize = arguments.batchSize if arguments.batchSize is not None else 2
kBatchSize = arguments.kBatchSize if arguments.kBatchSize is not None else 2
lossThreshold = arguments.lossThreshold if arguments.lossThreshold is not None else -1
dropoutRate = arguments.dropoutRate if arguments.dropoutRate is not None else -1
postLayerSizes = arguments.postLayerSizes if arguments.postLayerSizes is not None else [2]
preLayerSizes = arguments.preLayerSizes if arguments.preLayerSizes is not None else []
lstmLayerSizes = arguments.lstmLayerSizes if arguments.lstmLayerSizes is not None else [2]
c3dLayerSizes = arguments.c3dLayerSizes if arguments.c3dLayerSizes is not None else [2]
preLayerType = arguments.preLayerType if arguments.preLayerType is not None else None
postLayerType = arguments.postLayerType if arguments.postLayerType is not None else None
postLayerSize = str(postLayerSizes)
preLayerSize = str(preLayerSizes)
lstmLayerSize = str(lstmLayerSizes)
edgeDetection = arguments.edgeDetection if (arguments.edgeDetection is not None and arguments.edgeDetection >=0) else None
ageSelection = arguments.ageSelection if arguments.ageSelection is not None else 0
reluAlpha = arguments.reluAlpha if arguments.reluAlpha is not None else defaultReluAlpha
positiveSlicesThreshold = arguments.positiveSlicesThreshold if arguments.positiveSlicesThreshold is not None else -1
modelType = arguments.modelType if arguments.modelType is not None else 'vgg'
standardize = arguments.standardize if arguments.standardize is not None else False
trainOnValidationData = arguments.trainOnValidationData if arguments.trainOnValidationData is not None else False
numberOfReplications = arguments.replications if arguments.replications is not None else 1
usingLSTM = False
usingConv3D = False
usingResnet = False
if (modelType=='c3d'):
    usingConv3D = True
if (modelType == 'lstm'):
    usingLSTM = True
if (modelType == 'resnet'):
    usingResnet = True
channelCount = 3 if modelType == 'vgg' or (modelType == 'lstm' and preLayerType == 'vgg') else 1
stackedSlices = arguments.stackedSlices if arguments.stackedSlices is not None else False
trainingPatience = arguments.trainingPatience if arguments.trainingPatience is not None else 5
organParadigm = arguments.organParadigm if arguments.organParadigm is not None else 'separate'
plotLearning = arguments.plotLearning if arguments.plotLearning is not None else False
modelDescriptorString = str(int(round(time.time()*1000))) if (arguments.modelDescriptorString is None) else arguments.modelDescriptorString
baseDescriptorString = copy.copy(modelDescriptorString)
basePath = os.path.join(pathToIntermediateData,baseDescriptorString)
# neymanPearson = arguments.neymanPearson if arguments.neymanPearson is not None else -1
# kToLeaveOut = arguments.kToLeaveOut if arguments.kToLeaveOut is not None else 1
########################################################################################################################
#instantiate a container to aggregate the ROC curves (i.e. list of pairs {(f_i, t_i), where f_i and t_i are a simultaneous false and true positive rate respectively})
aggregateROCs = []
#repeate the experiment numberOfReplications times
for curRep in range(numberOfReplications):
    ########################################################################################################################
    #organize our shelf name
    modelDescriptorString = baseDescriptorString + '_' + str(curRep)
    shelfName = os.path.join(pathToIntermediateData, ''.join(['cd_shelf', modelDescriptorString, '.shelf']))#''.join(['cancer_detector_shelf', modelDescriptorString, '.shelf'])
    modelName = ''.join(['l3O', modelDescriptorString])
    modelPath = os.path.join(pathToIntermediateData, modelName)
    # os.makedirs(os.path.dirname(modelPath), exist_ok=True)
    # os.makedirs(os.path.dirname(shelfPath), exist_ok=True)
    #check if path exists and if not, create the directory
    if (not os.path.isdir(pathToIntermediateData)):
        try:
            os.mkdir(pathToIntermediateData)
        except:
            print('warning: intermediate data path %s may not exist'%str(pathToIntermediateData))
            print(pathToIntermediateData)
    #check if path exists and if not, create the directory
    if (not os.path.isdir(modelPath)):
        try:
            os.mkdir(modelPath)
        except:
            print('warning: intermediate data path %s may not exist'%str(modelPath))
            print(modelPath)

    #store this run's arguments in a text file in the intermediate data
    argsFileName = modelName+'_args.txt'
    argsFile = open(os.path.join(modelPath, argsFileName), "w")
    argsFile.write(str(arguments))
    argsFile.close()
    ########################################################################################################################
    print('organizing the data')
    ########################################################################################################################
    # organize the data
    inputs = None
    tm = TomographyManager.tomographyManager(pathToTomographyFiles, verbosity=verbosity)
    intervalSize = imageInterval[1]-imageInterval[0]+1


    # define a clipping eLU function
    def clipping_relu(x, alpha=reluAlpha):
        # pass through relu
        # y = K.relu(y, max_value=1)
        return tensorflow.clip_by_value(tensorflow.nn.elu(x),
                                        tensorflow.constant(-1.0),
                                        tensorflow.constant(1.0))


    chosenActivation = clipping_relu if reluAlpha is not None else "tanh"

    # optimization details
    learning_rate, learning_rate_drop = .001, 20  # .001, 20 try 0.0001

    #define a function to assembel a new model for us every time it's called
    def assembleModel(chosenActivation=chosenActivation, layersToFreeze=layersToFreeze, summary=False):
        # configure the model
        # sgd = tf.keras.optimizers.Nadam(learning_rate=learning_rate)  # Adadelta(learning_rate=learning_rate,  rho=0.95, epsilon=1e-06, decay=0.)
        #method to get the learning rate from the optimizer's automatic adaptation
        def lr_metric(optimizer):
            def lr(y_true, y_pred):
                return sgd.lr
            return lr
        # lr_metric = lr_metric(sgd)


        # optimization details
        # def lr_scheduler(epoch):
        #     return learning_rate * (0.5 ** (epoch // learning_rate_drop))
        # lr_scheduler = lr_scheduler
            # reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
        sgd = "adadelta" if usingLSTM else tf.keras.optimizers.Nadam(learning_rate=learning_rate)#tf.keras.optimizers.Adadelta(learning_rate=1.0)
        # reduceLRCallback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

        if (modelType == 'babyVGG'):
            inputLayer = Input(shape=(tm.imageRows, tm.imageColumns, 1))

            #downscale the image by 3x
            intermediate = inputLayer
            intermediate = tf.image.resize_with_pad(inputLayer, 128, 128)

            # define a clipping ReLU function
            def clipping_relu(x, alpha=reluAlpha):
                # pass through relu
                # y = K.relu(y, max_value=1)
                return tensorflow.clip_by_value(tensorflow.nn.elu(x),
                                                tensorflow.constant(-1.0),
                                                tensorflow.constant(alpha))

            # decide an activation function
            chosenActivation = clipping_relu if reluAlpha is not None else "tanh"

            #layer 0
            intermediate = Conv2D(64, (3,3), kernel_regularizer=regularizers.l2(0.0005), activation=chosenActivation, padding='same')(intermediate)
            intermediate = BatchNormalization()(intermediate)
            if (dropoutRate > 0):
                intermediate = Dropout(dropoutRate)(intermediate)

            # layer 1
            intermediate = Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.0005), activation=chosenActivation, padding='same')(intermediate)
            intermediate = BatchNormalization()(intermediate)

            # pooling
            # define a maxpooling layer between the encoder and decoder side of the CAE
            intermediate = MaxPool2D((2, 2), padding='same')(intermediate)
            if (dropoutRate > 0):
                intermediate = Dropout(dropoutRate)(intermediate)

            #transition to dense layers
            intermediate = Flatten()(intermediate)

            #add post-vgg-intermediate layers
            # layerCount = 0
            for curSize in postLayerSizes:
                if (dropoutRate != -1):
                    intermediate = Dropout(dropoutRate)(intermediate)
                intermediate = Dense(curSize, activation=chosenActivation,
                                     kernel_regularizer=regularizers.l2(0.0005))(intermediate)

            #prediction layer
            prediction = Dense(1, activation="sigmoid")(intermediate)

            #compile our model
            ourModel = Model(inputs=inputLayer, outputs=prediction)
            ourModel.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

        elif (modelType == 'vgg'):
            #set up vgg
            vggModel = VGG16(weights='imagenet', input_shape=(tm.imageRows,tm.imageColumns, channelCount), include_top=False)
            # vggModel.summary()

            #freeze the first layersToFreeze layers
            if (layersToFreeze == -1):
                layersToFreeze = len(vggModel.layers)
            for layer in vggModel.layers[:layersToFreeze]:
                layer.trainable = False

            # append an output processing and decoding layer
            intermediate = vggModel.output
            intermediate = Flatten()(intermediate)

            #add post-vgg-intermediate layers
            # layerCount = 0
            for curSize in postLayerSizes:
                if (dropoutRate != -1):
                    intermediate = Dropout(dropoutRate)(intermediate)
                intermediate = Dense(curSize, activation=chosenActivation,
                                     kernel_regularizer=regularizers.l2(0.0005))(intermediate)

            #prediction layer
            prediction = Dense(1, activation="sigmoid")(intermediate)

            #compile our model
            ourModel = Model(inputs=vggModel.input, outputs=prediction)
            ourModel.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

        elif (modelType == 'xception'):
            xceptionModel = Xception(include_top=False, weights='imagenet', input_shape=(tm.imageRows, tm.imageColumns, channelCount),
                                                 pooling=None, classes=2)

            # freeze the first layersToFreeze layers
            if (layersToFreeze == -1):
                layersToFreeze = len(xceptionModel.layers)
            for layer in xceptionModel.layers[:layersToFreeze]:
                layer.trainable = False

            # append an output processing and decoding layer
            intermediate = xceptionModel.output
            intermediate = Flatten()(intermediate)

            # add post-vgg-intermediate layers
            # layerCount = 0
            for curSize in postLayerSizes:
                # if layerCount == 0:
                #     intermediate = Dense(curSize, activation="tanh", activity_regularizer=regularizers.l2(0.0005))(intermediate)
                # else:
                intermediate = Dense(curSize, activation=chosenActivation, kernel_regularizer=regularizers.l2(0.0005))(
                    intermediate)
                if (dropoutRate != -1):
                    intermediate = Dropout(dropoutRate)(intermediate)

            # prediction layer
            prediction = Dense(2, activation="sigmoid")(intermediate)

            # compile our model
            # if gpuNameList is not None and len(gpuNameList) > 1:
            #     ourModel = Model(inputs=vggModel.input, outputs=prediction)
            #     ourModel.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
            #     ourModel = multi_gpu_model(ourModel, gpus=len(gpuNameList))
            #     ourModel.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
            # else:
            ourModel = Model(inputs=xceptionModel.input, outputs=prediction)
            ourModel.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

        elif (usingLSTM):

            def clipping_relu(x, alpha=reluAlpha):
                # pass through relu
                # y = K.relu(y, max_value=1)
                return tensorflow.clip_by_value(tensorflow.nn.elu(x),
                                                tensorflow.constant(-1.0),
                                                tensorflow.constant(1.0))
            #setup input
            # inputLayer = Input(shape=((2*intervalSize if organParadigm=='together' else intervalSize),
            #                           tm.imageRows, tm.imageColumns, channelCount))
            # intermediate = inputLayer
            # intermediate = TimeDistributed(intermediate)

            #setup preprocessing layers
            #rewrite the prelayers when I return
            #only single channel convlstm2ds are supported right now; return here in the future when we can have a separate model for
            #each channel and then have a concatenation layer feed into the post processing stack
            if('cnn' in preLayerType):
                inputLayer = Input(shape=((2*intervalSize if organParadigm=='together' else intervalSize), tm.imageRows, tm.imageColumns, 1))
                intermediate = inputLayer
                sizeIndex = 1
                for curSize in preLayerSizes:
                    intermediate = TimeDistributed(Conv2D(curSize, (3,3), kernel_regularizer=regularizers.l2(0.001),
                                                          padding='same', activation=chosenActivation))(intermediate)
                    if (dropoutRate != -1):
                        intermediate = Dropout(dropoutRate)(intermediate)
                    if (sizeIndex % 2 == 0):
                        intermediate = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(intermediate)
                    sizeIndex += 1
                # intermediate = TimeDistributed(Flatten())(intermediate)
                # # if (dropoutRate != -1):
                # #     intermediate = Dropout(dropoutRate)(intermediate)
                # # intermediate = Dense(preLayerSizes[-1], activation=chosenActivation, activity_regularizer=regularizers.l2(0.001))(intermediate)

                # setup stacked LSTM
                for curSize in lstmLayerSizes:
                    intermediate = Bidirectional(ConvLSTM2D(filters=curSize, kernel_size=(3, 3),
                                                            kernel_regularizer=regularizers.l2(0.001),
                                                            padding='same', return_sequences=True))(intermediate)
                    intermediate = BatchNormalization()(intermediate)
                    if (dropoutRate != -1):
                        intermediate = Dropout(dropoutRate)(intermediate)

                if (postLayerType == 'cnn'):
                    previousLayer = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(intermediate)
                    for curSize in postLayerSizes:
                        intermediate = TimeDistributed(Conv2D(curSize, (3, 3), kernel_regularizer=regularizers.l2(0.001),
                                                              padding='same', activation=chosenActivation))(intermediate)
                        intermediate = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(intermediate)
                        if (dropoutRate != -1):
                            intermediate = Dropout(dropoutRate)(intermediate)

                #prepare for a dense layer
                intermediate = TimeDistributed(Flatten())(intermediate)

            elif('mlp' in preLayerType):
                inputLayer = Input(shape=((2 * intervalSize if organParadigm == 'together' else intervalSize), tm.imageRows*tm.imageColumns))
                intermediate = inputLayer
                for curSize in preLayerSizes:
                    intermediate = TimeDistributed(Dense(curSize, activation=chosenActivation,
                                                         kernel_regularizer=regularizers.l2(0.001)))(intermediate)
                    if (dropoutRate != -1):
                        intermediate = TimeDistributed(Dropout(dropoutRate))(intermediate)

                #setup stacked LSTM
                for curSize in lstmLayerSizes:
                    intermediate = Bidirectional(LSTM(units=curSize, return_sequences=True))(intermediate)
                    intermediate = BatchNormalization()(intermediate)
                    if (dropoutRate != -1):
                        intermediate = Dropout(dropoutRate)(intermediate)

            if (postLayerType == 'mlp'):
                #post RNN layers
                for curSize in postLayerSizes:
                    intermediate = Dense(curSize, activation=chosenActivation, activity_regularizer=regularizers.l2(0.001))(intermediate)
                    if (dropoutRate != -1):
                        intermediate = Dropout(dropoutRate)(intermediate)

            #decision layer
            prediction = Dense(1, activation="sigmoid")(intermediate)

            #consider flattening the prediction sequence and applying a layer of logistic regression (ie dense())
            ourModel = Model(inputs=inputLayer, outputs=prediction)
            ourModel.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

            # double check weight trainability bug
            allVars = ourModel.variables
            trainableVars = ourModel.trainable_variables
            allVarNames = [ourModel.variables[i].name for i in range(len(ourModel.variables))]
            trainableVarNames = [ourModel.trainable_variables[i].name for i in range(len(ourModel.trainable_variables))]
            nonTrainableVars = np.setdiff1d(allVarNames, trainableVarNames)

            if (len(nonTrainableVars) > 0):
                print('the following variables are set to non-trainable; ensure that this is correct before publishing!!!!')
                print(nonTrainableVars)

        elif (usingConv3D):

            def clipping_relu(x, alpha=reluAlpha):
                # pass through relu
                # y = K.relu(y, max_value=1)
                return tensorflow.clip_by_value(tensorflow.nn.elu(x),
                                                tensorflow.constant(-1.0),
                                                tensorflow.constant(1.0))
            #setup input
            # inputLayer = Input(shape=((2*intervalSize if organParadigm=='together' else intervalSize),
            #                           tm.imageRows, tm.imageColumns, channelCount))
            # intermediate = inputLayer
            # intermediate = TimeDistributed(intermediate)

            #setup preprocessing layers
            #rewrite the prelayers when I return
            #only single channel convlstm2ds are supported right now; return here in the future when we can have a separate model for
            #each channel and then have a concatenation layer feed into the post processing stack
            if('cnn' in preLayerType):
                inputLayer = Input(shape=((2*intervalSize if organParadigm=='together' else intervalSize), tm.imageRows, tm.imageColumns, 1))
                intermediate = inputLayer
                sizeIndex = 1
                for curSize in preLayerSizes:
                    intermediate = TimeDistributed(Conv2D(curSize, (3,3), kernel_regularizer=regularizers.l2(0.0005),
                                                          padding='same', activation=chosenActivation))(intermediate)
                    if (dropoutRate != -1):
                        intermediate = Dropout(dropoutRate)(intermediate)
                    # if (sizeIndex % 2 == 0):
                    #     intermediate = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(intermediate)
                    intermediate = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(intermediate)
                    sizeIndex += 1
                if (dropoutRate != -1):
                    intermediate = Dropout(dropoutRate)(intermediate)
                # intermediate = TimeDistributed(Flatten())(intermediate)
                # # if (dropoutRate != -1):
                # #     intermediate = Dropout(dropoutRate)(intermediate)
                # # intermediate = Dense(preLayerSizes[-1], activation=chosenActivation, activity_regularizer=regularizers.l2(0.0005))(intermediate)

                # setup 3d cnn
                for curSize in c3dLayerSizes:
                    intermediate = Conv3D(filters=curSize, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                          kernel_regularizer=regularizers.l2(0.0005),
                                          padding='same')(intermediate)
                    intermediate = BatchNormalization()(intermediate)
                    if (dropoutRate != -1):
                        intermediate = Dropout(dropoutRate)(intermediate)

                if (postLayerType == 'cnn'):
                    previousLayer = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(intermediate)
                    for curSize in postLayerSizes:
                        intermediate = TimeDistributed(Conv2D(curSize, (3, 3), kernel_regularizer=regularizers.l2(0.0005),
                                                              padding='same', activation=chosenActivation))(intermediate)
                        if (dropoutRate != -1):
                            intermediate = Dropout(dropoutRate)(intermediate)
                        intermediate = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))(intermediate)

                #prepare for a dense layer
                intermediate = TimeDistributed(Flatten())(intermediate)

            if (postLayerType == 'mlp'):
                #post RNN layers
                if (dropoutRate != -1):
                    intermediate = Dropout(dropoutRate)(intermediate)
                for curSize in postLayerSizes:
                    intermediate = Dense(curSize, activation=chosenActivation, kernel_regularizer=regularizers.l2(0.0005))(intermediate)
                    if (dropoutRate != -1):
                        intermediate = Dropout(dropoutRate)(intermediate)

            #decision layer
            prediction = Dense(1, activation="sigmoid")(intermediate)

            #consider flattening the prediction sequence and applying a layer of logistic regression (ie dense())
            ourModel = Model(inputs=inputLayer, outputs=prediction)
            ourModel.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

        elif (modelType == 'mlp'):
            print("don't use an mlp")
            # # define a clipping ReLU function
            # def clipping_relu(x, alpha=reluAlpha):
            #     # pass through relu
            #     # y = K.relu(y, max_value=1)
            #     return tensorflow.clip_by_value(tensorflow.nn.elu(x),
            #                                     tensorflow.constant(-1.0),
            #                                     tensorflow.constant(1.0))
            #
            #
            # chosenActivation = clipping_relu if reluAlpha is not None else "tanh"
            #
            # # add post-vgg-intermediate layers
            # # layerCount = 0
            # inputLayer = Input(shape=(tm.imageRows*tm.imageColumns, ))
            # intermediate = inputLayer
            # for curSize in postLayerSizes:
            #     # if layerCount == 0:
            #     #     intermediate = Dense(curSize, activation="tanh", activity_regularizer=regularizers.l2(0.0005))(intermediate)
            #     # else:
            #     intermediate = Dense(curSize, activation=chosenActivation, activity_regularizer=regularizers.l2(0.0005))(
            #         intermediate)
            #     if (dropoutRate != -1):
            #         intermediate = Dropout(dropoutRate)(intermediate)
            #
            # #average pooling
            # # intermediate = AveragePooling1D(pool_size=curSize)(intermediate)
            # # prediction layer
            # prediction = Dense(1, activation="sigmoid")(intermediate)
            #
            # # compile our model
            # # if gpuNameList is not None and len(gpuNameList) > 1:
            # #     ourModel = Model(inputs=vggModel.input, outputs=prediction)
            # #     ourModel.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
            # #     ourModel = multi_gpu_model(ourModel, gpus=len(gpuNameList))
            # #     ourModel.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
            # # else:
            # ourModel = Model(inputs=inputLayer, outputs=prediction)
            # ourModel.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

        elif (modelType == 'resnet'):
            #look at old code if you want to use the pretrained resnet
            #I'm thinking that a residual network with a 3d cnn in the middle connecting to a 1d-maxpooling (over time) into a
            #cnn (to which the residual connections connect) -> MLP
            def clipping_relu(x, alpha=reluAlpha):
                # pass through relu
                # y = K.relu(y, max_value=1)
                return tensorflow.clip_by_value(tensorflow.nn.elu(x),
                                                tensorflow.constant(-1.0),
                                                tensorflow.constant(1.0))
            sequenceLength = (2*intervalSize if organParadigm=='together' else intervalSize)
            inputLayer = Input(shape=(sequenceLength, tm.imageRows, tm.imageColumns, 1))
            intermediate = inputLayer
            #pad the 2nd spatial dimension with 1 column so we get 376 by 376
            intermediate = TimeDistributed(tf.keras.layers.ZeroPadding2D(padding=((0,0),(0,2))))(intermediate)
            #downscale the images
            intermediate = TimeDistributed(tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (256, 256))))(intermediate)

            sizeIndex = 1
            residualConnections = dict()
            for curSize in preLayerSizes:
                intermediate = TimeDistributed(Conv2D(curSize, (3,3), kernel_regularizer=regularizers.l2(0.0005),
                                                      padding='same', activation=chosenActivation))(intermediate)
                residualConnections[sizeIndex] = BatchNormalization()(intermediate)
                intermediate = TimeDistributed(MaxPool2D((2, 2), padding='same'))(residualConnections[sizeIndex])
                if (dropoutRate != -1):
                    intermediate = Dropout(dropoutRate)(intermediate)

                # intShape = tf.shape(intermediate)
                # reshaped = tf.reshape(intermediate, (-1, intShape[1], intShape[2]*intShape[3]*intShape[4]))
                # interReshaped = tf.keras.layers.GlobalMaxPooling1D(data_format="channels_last")(reshaped)
                # residualConnections[sizeIndex] = tf.reshape(interReshaped, (-1, intShape[1], intShape[2], intShape[3], intShape[4]))
                sizeIndex += 1

            # setup 3d cnn
            for curSize in c3dLayerSizes:
                intermediate = Conv3D(filters=curSize, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                      kernel_regularizer=regularizers.l2(0.0005),
                                      padding='same')(intermediate)
                intermediate = BatchNormalization()(intermediate)
                if (dropoutRate != -1):
                    intermediate = Dropout(dropoutRate)(intermediate)
            sizeIndex -= 1
            #connect up residual wires
            for curSize in postLayerSizes[:(len(postLayerSizes)-len(preLayerSizes))]:
                intermediate = TimeDistributed(tf.keras.layers.UpSampling2D((2, 2), interpolation='nearest'))(intermediate)

                intermediate = tf.keras.layers.concatenate([intermediate, residualConnections[sizeIndex]], axis=4)
                intermediate = TimeDistributed(tf.keras.layers.Conv2DTranspose(curSize, kernel_size=(3, 3),
                                kernel_regularizer=regularizers.l2(0.0005),
                                padding='same', activation=chosenActivation))(intermediate)
                sizeIndex-=1

            #downsample in space
            intermediate = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=None, padding="valid")(intermediate)

            #prepare for MLP
            intermediate = TimeDistributed(Flatten())(intermediate)
            if (dropoutRate != -1):
                intermediate = Dropout(dropoutRate)(intermediate)

            #connect up the mlp
            for curSize in postLayerSizes[(len(preLayerSizes)+1):]:
                intermediate = Dense(curSize, activation=chosenActivation, kernel_regularizer=regularizers.l2(0.0005))(intermediate)
                if (dropoutRate != -1):
                    intermediate = Dropout(dropoutRate)(intermediate)

            # decision layer
            prediction = Dense(1, activation="sigmoid")(intermediate)

            # consider flattening the prediction sequence and applying a layer of logistic regression (ie dense())
            ourModel = Model(inputs=inputLayer, outputs=prediction)
            ourModel.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

        if (verbosity):
            print('model compiled')
        if (summary):
            ourModel.summary()

        return ourModel, ourModel.optimizer

    ourModel, sgd = assembleModel(chosenActivation=chosenActivation, summary=True)
    #store the initial weights for our cross validation experiment
    initialWeights = copy.deepcopy(ourModel.get_weights())

    # collect a list of animal ids
    #for debugging: np.array([3624, 3316]).astype(int)
    # allAnimalIDs = np.array([3624, 3316, 3551, 3761]).astype(int)
    # allAnimalIDs = tm.getAllAnimalIDs().astype(int)
    if (ageSelection == -1):
        allAnimalIDs = tm.getAllAnimalIDs()
    else:
        allAnimalIDs = tm.getAnimalIDsByAge(ageSelection)

    #assemble the list of [(trainingID, testingID)] to split the dataset
    # random.shuffle(allAnimalIDs)
    numberOfFolds = len(allAnimalIDs)
    loo = LeaveOneOut()
    foldIndex, confusions = 1, dict()
    testPredictions, groundTruth = np.array([]), []
    numberOfFolds = loo.get_n_splits(allAnimalIDs)

    animalIDList, predictionProbabilityList = [], []
    for trainingIDs, testingIDs in loo.split(allAnimalIDs):
        #shuffle trainingIDs
        # random.shuffle(trainingIDs)

        #select holdout validation IDs
        trainingIDsY = np.array(tm.getLabelsByAnimalID(allAnimalIDs[trainingIDs]))
        trainingIDsPos, trainingIDsNeg = allAnimalIDs[trainingIDs[np.where(trainingIDsY == 1)[0]]],\
                                         allAnimalIDs[trainingIDs[np.where(trainingIDsY == 0)[0]]]
        # posValID = random.sample(list(trainingIDsPos), 1)[0]#rng set once at the beginning so this should be consistent across multiple runs of the same experiment
        # validationIDs = [posValID]
        # negValID = random.sample(list(trainingIDsNeg), 1)[0]#rng set once at the beginning so this should be consistent across multiple runs of the same experiment
        # validationIDs.append(negValID)
        # validationIndices = tm.getIndicesFromAnimalIDs(validationIDs, imageInterval=imageInterval)

        #aggregate our training indices
        # trainingIDsWithoutValidationIDs = np.setdiff1d(allAnimalIDs[trainingIDs], validationIDs)
        # posNoVals = trainingIDsPos.copy().tolist()
        # posNoVals.remove(posValID)
        # # posNoVa ls = np.delete(posNoVals, posValID)
        # negNoVals = trainingIDsNeg.copy().tolist()
        # negNoVals.remove(negValID)
        # # negNoVals = np.delete(negNoVals, negValID)
        # #stratify training IDs (more important for lstm and C3D but why not do it for all methods)
        stratifiedTrainingIDsNoVal = [id for pair in zip(trainingIDsPos, trainingIDsNeg) for id in pair if id is not None]
        stratifiedLabels = [tm.getLabelsByAnimalID([curID]) for curID in stratifiedTrainingIDsNoVal]
        print(stratifiedLabels)
        trainingIndices = tm.getIndicesFromAnimalIDs(stratifiedTrainingIDsNoVal, imageInterval=imageInterval)

        #if we're normalizing the data, calculate the stats here
        if (standardize):
            (pixelMean, pixelSTD) = tm.standardFit(trainingIndices, cannySigma=edgeDetection, preLayerType=preLayerType,
                                                   modelType=modelType)
        # #read in validation data
        # validationX, validationY = tm.dataSummoner(validationIndices, gaussianFilterSigma=gaussianFilterSigma,
        #                                            preprocessingFilterDimensionality=preprocessingFilterDimensionality,
        #                                            cannySigma=edgeDetection, standardize=standardize,
        #                                            modelType=modelType, numberOfChannels=channelCount, preLayerType=preLayerType,
        #                                            stackedSlices=stackedSlices, numberOfSequences=((4 if organParadigm=='separate' else 2) if (usingLSTM or usingConv3D or usingResnet) else None),
        #                                            slicesPerSequence = (2*intervalSize if organParadigm=='together' else intervalSize) if (usingLSTM or usingConv3D or usingResnet) else None)
        # validationTargets = utils.to_categorical(validationY, num_classes=2)
        # trainingIndices = trainingIndices[:5]
        # split training and testing indices across batches
        # if we're using an LSTM, ensure that batchSize is the nearest multiple of sequenceLength
        # also if we're using organparadigm=='together', then we need this to be a multiple of 2
        #todo: exception hadnling for the above note?
        trainingIndicesBatches = tm.splitIndicesAcrossBatches(trainingIndices, batchSize, shuffle=False)
        if (verbosity):
            print('fold %i of %i' %(foldIndex, numberOfFolds))
            print('commencing training')

        # train for specified number of epochs and stop early if we reach a good fit
        # note that early stopping restore_best_weights only restores the best weights
        # when training stops early (see https://github.com/keras-team/keras/issues/12511)
        # so if we want to reliably restore the best weights, we need to use the model checkpoint
        import datetime
        earlyStopper = EarlyStopping(monitor='val_loss', mode='min', patience=trainingPatience,
                                     verbose=1, min_delta=lossThreshold)
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tbCallback = tf.keras.callbacks.TensorBoard(log_dir='graphs', histogram_freq=1)

        checkpoint = tf.compat.v1.keras.callbacks.ModelCheckpoint(os.path.join(modelPath,'bestModelLastIt.h5'), verbose=1, monitor='val_loss',
                                     save_weights_only=True,
                                     save_best_only=True, mode='auto')
        # iterate over training batches and fit the model
        if (plotLearning):
            aggHistory = dict()
            aggHistory['val_loss'], aggHistory['loss'], aggHistory['acc'], aggHistory['val_acc'] = [], [], [], []

        trainingBatch = 0
        trainingComplete = False
        curEffectiveEpoch = 1
        for curTrainingIndices in trainingIndicesBatches:
            trainingBatch += 1
            # collect this iteration's data and labels
            nextTrainingIndices = trainingIndicesBatches[trainingBatch] if trainingBatch < len(trainingIndicesBatches) else trainingIndicesBatches[0]
            sequencesInBatch = (int)(len(curTrainingIndices)/((2 if organParadigm == 'together' else 1)*intervalSize))
            sequencesInNextBatch = (int)(len(nextTrainingIndices)/((2 if organParadigm == 'together' else 1)*intervalSize))
            if (sequencesInBatch == 0 and (usingLSTM or usingConv3D)):
                print("rethink the batch sizing: sequencesInBatch rounded to 0")
            if (verbosity):
                print('summoning data in batch %i' % trainingBatch)


            validationX, validationY = tm.dataSummoner(nextTrainingIndices, gaussianFilterSigma=gaussianFilterSigma,
                                                       preprocessingFilterDimensionality=preprocessingFilterDimensionality,
                                                       modelType=modelType, numberOfChannels=channelCount,
                                                       stackedSlices=stackedSlices, cannySigma=edgeDetection,
                                                       standardize=standardize,
                                                       numberOfSequences=sequencesInNextBatch if (
                                                               usingLSTM or usingConv3D or usingResnet) else None,
                                                       preLayerType=preLayerType,
                                                       slicesPerSequence=(2 * intervalSize if organParadigm == 'together' else intervalSize) if (usingLSTM or usingConv3D or usingResnet) else None)
            trainingX, trainingY = tm.dataSummoner(curTrainingIndices, gaussianFilterSigma=gaussianFilterSigma,
                                                   preprocessingFilterDimensionality=preprocessingFilterDimensionality,
                                                   modelType=modelType, numberOfChannels=channelCount,
                                                   stackedSlices=stackedSlices, cannySigma=edgeDetection,
                                                   standardize=standardize, numberOfSequences=sequencesInBatch if (usingLSTM or usingConv3D or usingResnet) else None,
                                                   preLayerType=preLayerType,
                                                   slicesPerSequence=(2*intervalSize if organParadigm=='together' else intervalSize) if (usingLSTM or usingConv3D or usingResnet) else None)
            print('labels in batch %i: %s' % (trainingBatch, str(np.unique(trainingY))))


            # convert targets to categorical encoding
            trainingTargets = np.array(trainingY)
            validationTargets = np.array(validationY)
            # trainingTargets = np.array([utils.to_categorical(trainingY[i], num_classes=2) for i in range(len(trainingY))])
            # validationTargets = np.array([utils.to_categorical(validationY[i], num_classes=2) for i in range(len(validationY))])

            if (usingLSTM or usingConv3D or usingResnet):
                curHist = ourModel.fit(trainingX, trainingTargets, validation_data=(validationX, validationTargets),
                                       epochs=fineTuningEpochs, verbose=1,
                                       batch_size=kBatchSize, callbacks=[earlyStopper, checkpoint])


                if (verbosity):
                    bestLossId = np.argmin(curHist.history['val_loss'])
                    print('final validation loss: %f' % (curHist.history['val_loss'][bestLossId]))
                    if ('val_acc' in list(curHist.history.keys())):
                        print('final validation acc: %f' % ((curHist.history['val_acc'][bestLossId])))
                    else:
                        print('final validation acc: %f' % (curHist.history['val_accuracy'][bestLossId]))
            else:
                curHist = ourModel.fit(trainingX, trainingTargets, validation_data=(validationX, validationTargets),
                                       epochs=fineTuningEpochs, verbose=1,
                                       batch_size=kBatchSize, callbacks=[earlyStopper, checkpoint])
                if (verbosity):
                    bestLossId = np.argmin(curHist.history['val_loss'])
                    print('final validation loss: %f' % (curHist.history['val_loss'][bestLossId]))
                    if ('val_acc' in list(curHist.history.keys())):
                        print('final validation acc: %f' % ((curHist.history['val_acc'][bestLossId])))
                    else:
                        print('final validation acc: %f' % (curHist.history['val_accuracy'][bestLossId]))

            if (plotLearning):
                aggHistory['loss'].extend(curHist.history['loss'][:])
                aggHistory['val_loss'].extend(curHist.history['val_loss'][:])
                if ('val_acc' in list(curHist.history.keys())):
                    aggHistory['val_acc'].extend(curHist.history['val_acc'][:])
                else:
                    aggHistory['val_acc'].extend(curHist.history['val_accuracy'][:])
                if ('acc' in list(curHist.history.keys())):
                    aggHistory['acc'].extend(curHist.history['acc'][:])
                else:
                    aggHistory['acc'].extend(curHist.history['accuracy'][:])
                fig, (axL, axA) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
                # fig.set_figwidth(16)
                # fig.set_figheight(6)
                historyLength = len(aggHistory['loss'])
                axL.plot(range(historyLength), np.array(aggHistory['loss']).reshape(-1,1), linestyle='-.', label='training')
                axL.plot(range(historyLength), np.array(aggHistory['val_loss']).reshape(-1,1), linestyle='--', label='validation')
                axL.set(xlabel='epoch', ylabel='loss')
                axA.plot(range(historyLength), np.array(aggHistory['acc']).reshape(-1,1), linestyle='-.', label='training')
                axA.plot(range(historyLength), np.array(aggHistory['val_acc']).reshape(-1,1), linestyle='--', label='validation')
                axA.set(xlabel='epoch', ylabel='accuracy')
                plt.legend()
                plt.tight_layout()
                fig.subplots_adjust(top=.9)
                plt.suptitle('Performance throughout learning')
                plt.savefig(os.path.join(modelPath,'perf_in_training_tb_'+str(trainingBatch)+'.png'))
                plt.show()

                fig, ax1 = plt.subplots(nrows=1, ncols=1)
                fig.suptitle('Performance throughout learning')
                ax2 = ax1.twinx()
                ax1.plot(range(historyLength), np.array(aggHistory['loss']).reshape(-1,1), marker='o', linestyle='-.', label='training')
                ax1.plot(range(historyLength), np.array(aggHistory['val_loss']).reshape(-1,1), marker='o', linestyle='--', label='validation')
                ax2.plot(range(historyLength), np.array(aggHistory['acc']).reshape(-1,1), marker='.', linestyle='-.', label='training')
                ax2.plot(range(historyLength), np.array(aggHistory['val_acc']).reshape(-1,1), marker='.', linestyle='--', label='validation')
                plt.legend()
                plt.savefig(os.path.join(modelPath,'single_perf_in_training_tb_'+str(trainingBatch)+'.png'))
                plt.show()

            #load the best model
            ourModel.load_weights(os.path.join(modelPath,'bestModelLastIt.h5'))
            # restore learning rate corresponding to best model
            curEffectiveEpoch+=np.argmin(curHist.history['val_loss'])
            # sgd.lr.assign(lr_scheduler(curEffectiveEpoch))

            # free up some memory
            del trainingX
            del trainingY
            del trainingTargets

        #before we test anything, we can train directly on the validation data since this isn't truly held out, and just controls stopping on the remainder of the training data for this fold
        # train for specified number of epochs and stop early if we reach a good fit
        # note that early stopping restore_best_weights only restores the best weights
        # when training stops early (see https://github.com/keras-team/keras/issues/12511)
        # so if we want to reliably restore the best weights, we need to use the model checkpoint
        #cut the learning rate here to minimize divergence
        # if (trainOnValidationData):#eventually make this an argument?
        #     sgd.lr.assign(K.eval(sgd.lr)/10)
        #     # todo: use livelossplot to generate a plot of losses and accuracies of training vgg?
        #     #get pos and neg validation indices
        #
        #     if (usingLSTM or usingConv3D):
        #         firstEntry = np.argmax(validationTargets[0, 0, :])
        #         if (firstEntry == 1):
        #             posIndex = 0
        #             negIndex = 1
        #         else:
        #             posIndex = 1
        #             negIndex = 0
        #         #pos validation tuning
        #         curHist = ourModel.fit(np.array([validationX[posIndex, :, :, :, :]]), np.array([validationTargets[posIndex, :, :]]) ,
        #                                validation_data=(validationX, validationTargets),
        #                                epochs=fineTuningEpochs, verbose=1,
        #                                batch_size=kBatchSize, callbacks=[earlyStopper, checkpoint])
        #         # load the best model
        #         ourModel.load_weights(os.path.join(modelPath, 'bestModelLastIt.h5'))
        #         curEffectiveEpoch += np.argmin(curHist.history['val_loss'])
        #
        #         if (plotLearning):
        #             aggHistory['loss'].extend(curHist.history['loss'][:])
        #             aggHistory['val_loss'].extend(curHist.history['val_loss'][:])
        #             if ('val_acc' in list(curHist.history.keys())):
        #                 aggHistory['val_acc'].extend(curHist.history['val_acc'][:])
        #             else:
        #                 aggHistory['val_acc'].extend(curHist.history['val_accuracy'][:])
        #             if ('acc' in list(curHist.history.keys())):
        #                 aggHistory['acc'].extend(curHist.history['acc'][:])
        #             else:
        #                 aggHistory['acc'].extend(curHist.history['accuracy'][:])
        #             fig, (axL, axA) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        #             # fig.set_figwidth(16)
        #             # fig.set_figheight(6)
        #             historyLength = len(aggHistory['loss'])
        #             axL.plot(range(historyLength), np.array(aggHistory['loss']).reshape(-1,1), linestyle='-.', label='training')
        #             axL.plot(range(historyLength), np.array(aggHistory['val_loss']).reshape(-1,1), linestyle='--', label='validation')
        #             axL.set(xlabel='epoch', ylabel='loss')
        #             axA.plot(range(historyLength), np.array(aggHistory['acc']).reshape(-1,1), linestyle='-.', label='training')
        #             axA.plot(range(historyLength), np.array(aggHistory['val_acc']).reshape(-1,1), linestyle='--', label='validation')
        #             axA.set(xlabel='epoch', ylabel='accuracy')
        #             plt.legend()
        #             plt.tight_layout()
        #             fig.subplots_adjust(top=.9)
        #             plt.suptitle('Performance throughout learning')
        #             plt.savefig(os.path.join(modelPath,'perf_in_training_tb_'+str(trainingBatch)+'.png'))
        #             plt.show()
        #
        #             fig, ax1 = plt.subplots(nrows=1, ncols=1)
        #             fig.suptitle('Performance throughout learning')
        #             ax2 = ax1.twinx()
        #             ax1.plot(range(historyLength), np.array(aggHistory['loss']).reshape(-1,1), marker='o', linestyle='-.', label='training')
        #             ax1.plot(range(historyLength), np.array(aggHistory['val_loss']).reshape(-1,1), marker='o', linestyle='--', label='validation')
        #             ax2.plot(range(historyLength), np.array(aggHistory['acc']).reshape(-1,1), marker='.', linestyle='-.', label='training')
        #             ax2.plot(range(historyLength), np.array(aggHistory['val_acc']).reshape(-1,1), marker='.', linestyle='--', label='validation')
        #             plt.legend()
        #             plt.savefig(os.path.join(modelPath,'single_perf_in_training_tb_'+str(trainingBatch)+'.png'))
        #             plt.show()
        #         # restore learning rate corresponding to best model
        #         # sgd.lr.assign(lr_scheduler(curEffectiveEpoch))
        #
        #         if (verbosity):
        #             bestLossId = np.argmin(curHist.history['val_loss'])
        #             print('final validation loss: %f' % (curHist.history['val_loss'][bestLossId]))
        #             if ('val_acc' in list(curHist.history.keys())):
        #                 print('final validation acc: %f' % ((curHist.history['val_acc'][bestLossId])))
        #             else:
        #                 print('final validation acc: %f' % (curHist.history['val_accuracy'][bestLossId]))
        #
        #         #neg pass
        #         curHist = ourModel.fit(np.array([validationX[negIndex, :, :, :, :]]), np.array([validationTargets[negIndex, :, :]]),
        #                                epochs=fineTuningEpochs, verbose=1,
        #                                validation_data=(validationX, validationTargets),
        #                                batch_size=kBatchSize, callbacks=[earlyStopper, checkpoint])
        #         # load the best model
        #         ourModel.load_weights(os.path.join(modelPath, 'bestModelLastIt.h5'))
        #         curEffectiveEpoch += np.argmin(curHist.history['val_loss'])
        #         # restore learning rate corresponding to best model
        #         # sgd.lr.assign(lr_scheduler(curEffectiveEpoch))
        #         if (plotLearning):
        #             aggHistory['loss'].extend(curHist.history['loss'][:])
        #             aggHistory['val_loss'].extend(curHist.history['val_loss'][:])
        #             if ('val_acc' in list(curHist.history.keys())):
        #                 aggHistory['val_acc'].extend(curHist.history['val_acc'][:])
        #             else:
        #                 aggHistory['val_acc'].extend(curHist.history['val_accuracy'][:])
        #             if ('acc' in list(curHist.history.keys())):
        #                 aggHistory['acc'].extend(curHist.history['acc'][:])
        #             else:
        #                 aggHistory['acc'].extend(curHist.history['accuracy'][:])
        #             fig, (axL, axA) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        #             # fig.set_figwidth(16)
        #             # fig.set_figheight(6)
        #             historyLength = len(aggHistory['loss'])
        #             axL.plot(range(historyLength), np.array(aggHistory['loss']).reshape(-1,1), linestyle='-.', label='training')
        #             axL.plot(range(historyLength), np.array(aggHistory['val_loss']).reshape(-1,1), linestyle='--', label='validation')
        #             axL.set(xlabel='epoch', ylabel='loss')
        #             axA.plot(range(historyLength), np.array(aggHistory['acc']).reshape(-1,1), linestyle='-.', label='training')
        #             axA.plot(range(historyLength), np.array(aggHistory['val_acc']).reshape(-1,1), linestyle='--', label='validation')
        #             axA.set(xlabel='epoch', ylabel='accuracy')
        #             plt.legend()
        #             plt.tight_layout()
        #             fig.subplots_adjust(top=.9)
        #             plt.suptitle('Performance throughout learning')
        #             plt.savefig(os.path.join(modelPath,'perf_in_training_tb_'+str(trainingBatch)+'.png'))
        #             plt.show()
        #
        #             fig, ax1 = plt.subplots(nrows=1, ncols=1)
        #             fig.suptitle('Performance throughout learning')
        #             ax2 = ax1.twinx()
        #             ax1.plot(range(historyLength), np.array(aggHistory['loss']).reshape(-1,1), marker='o', linestyle='-.', label='training')
        #             ax1.plot(range(historyLength), np.array(aggHistory['val_loss']).reshape(-1,1), marker='o', linestyle='--', label='validation')
        #             ax2.plot(range(historyLength), np.array(aggHistory['acc']).reshape(-1,1), marker='.', linestyle='-.', label='training')
        #             ax2.plot(range(historyLength), np.array(aggHistory['val_acc']).reshape(-1,1), marker='.', linestyle='--', label='validation')
        #             plt.legend()
        #             plt.savefig(os.path.join(modelPath,'single_perf_in_training_tb_'+str(trainingBatch)+'.png'))
        #             plt.show()
        #
        #         if (verbosity):
        #             bestLossId = np.argmin(curHist.history['val_loss'])
        #             print('final validation loss: %f' % (curHist.history['val_loss'][bestLossId]))
        #             if ('val_acc' in list(curHist.history.keys())):
        #                 print('final validation acc: %f' % ((curHist.history['val_acc'][bestLossId])))
        #             else:
        #                 print('final validation acc: %f' % (curHist.history['val_accuracy'][bestLossId]))
        #     else:
        #         posIndices, negIndices = np.where(np.argmax(validationTargets, axis=1) == 1)[0], np.where(np.argmax(validationTargets, axis=1) == 0)[0]
        #         #pos pass
        #         curHist = ourModel.fit(validationX[posIndices, :, :], validationTargets[posIndices, :],
        #                                epochs=fineTuningEpochs, verbose=1,
        #                                validation_data=(validationX, validationTargets),
        #                                batch_size=kBatchSize, callbacks=[earlyStopper, checkpoint])
        #         # load the best model
        #         ourModel.load_weights(os.path.join(modelPath, 'bestModelLastIt.h5'))
        #         curEffectiveEpoch += np.argmin(curHist.history['val_loss'])
        #
        #         if (plotLearning):
        #             aggHistory['loss'].extend(curHist.history['loss'][:])
        #             aggHistory['val_loss'].extend(curHist.history['val_loss'][:])
        #             if ('val_acc' in list(curHist.history.keys())):
        #                 aggHistory['val_acc'].extend(curHist.history['val_acc'][:])
        #             else:
        #                 aggHistory['val_acc'].extend(curHist.history['val_accuracy'][:])
        #             if ('acc' in list(curHist.history.keys())):
        #                 aggHistory['acc'].extend(curHist.history['acc'][:])
        #             else:
        #                 aggHistory['acc'].extend(curHist.history['accuracy'][:])
        #             fig, (axL, axA) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        #             # fig.set_figwidth(16)
        #             # fig.set_figheight(6)
        #             historyLength = len(aggHistory['loss'])
        #             axL.plot(range(historyLength), np.array(aggHistory['loss']).reshape(-1,1), linestyle='-.', label='training')
        #             axL.plot(range(historyLength), np.array(aggHistory['val_loss']).reshape(-1,1), linestyle='--', label='validation')
        #             axL.set(xlabel='epoch', ylabel='loss')
        #             axA.plot(range(historyLength), np.array(aggHistory['acc']).reshape(-1,1), linestyle='-.', label='training')
        #             axA.plot(range(historyLength), np.array(aggHistory['val_acc']).reshape(-1,1), linestyle='--', label='validation')
        #             axA.set(xlabel='epoch', ylabel='accuracy')
        #             plt.legend()
        #             plt.tight_layout()
        #             fig.subplots_adjust(top=.9)
        #             plt.suptitle('Performance throughout learning')
        #             plt.savefig(os.path.join(modelPath,'perf_in_training_tb_'+str(trainingBatch)+'.png'))
        #             plt.show()
        #
        #             fig, ax1 = plt.subplots(nrows=1, ncols=1)
        #             fig.suptitle('Performance throughout learning')
        #             ax2 = ax1.twinx()
        #             ax1.plot(range(historyLength), np.array(aggHistory['loss']).reshape(-1,1), marker='o', linestyle='-.', label='training')
        #             ax1.plot(range(historyLength), np.array(aggHistory['val_loss']).reshape(-1,1), marker='o', linestyle='--', label='validation')
        #             ax2.plot(range(historyLength), np.array(aggHistory['acc']).reshape(-1,1), marker='.', linestyle='-.', label='training')
        #             ax2.plot(range(historyLength), np.array(aggHistory['val_acc']).reshape(-1,1), marker='.', linestyle='--', label='validation')
        #             plt.legend()
        #             plt.savefig(os.path.join(modelPath,'single_perf_in_training_tb_'+str(trainingBatch)+'.png'))
        #             plt.show()
        #
        #         # restore learning rate corresponding to best model
        #         # sgd.lr.assign(curHist.history['lr'][np.argmin(curHist.history['val_loss'])])
        #
        #         if (verbosity):
        #             bestLossId = np.argmin(curHist.history['val_loss'])
        #             print('final validation loss: %f' % (curHist.history['val_loss'][bestLossId]))
        #             if ('val_acc' in list(curHist.history.keys())):
        #                 print('final validation acc: %f' % ((curHist.history['val_acc'][bestLossId])))
        #             else:
        #                 print('final validation acc: %f' % (curHist.history['val_accuracy'][bestLossId]))
        #
        #         #neg pass
        #         curHist = ourModel.fit(validationX[negIndices, :, :], validationTargets[negIndices, :],
        #                                epochs=fineTuningEpochs, verbose=1,
        #                                validation_data=(validationX, validationTargets),
        #                                batch_size=kBatchSize, callbacks=[earlyStopper, checkpoint])
        #         # load the best model
        #         ourModel.load_weights(os.path.join(modelPath, 'bestModelLastIt.h5'))
        #         curEffectiveEpoch += np.argmin(curHist.history['val_loss'])
        #
        #         if (plotLearning):
        #             aggHistory['loss'].extend(curHist.history['loss'][:])
        #             aggHistory['val_loss'].extend(curHist.history['val_loss'][:])
        #             if ('val_acc' in list(curHist.history.keys())):
        #                 aggHistory['val_acc'].extend(curHist.history['val_acc'][:])
        #             else:
        #                 aggHistory['val_acc'].extend(curHist.history['val_accuracy'][:])
        #             if ('acc' in list(curHist.history.keys())):
        #                 aggHistory['acc'].extend(curHist.history['acc'][:])
        #             else:
        #                 aggHistory['acc'].extend(curHist.history['accuracy'][:])
        #             fig, (axL, axA) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        #             # fig.set_figwidth(16)
        #             # fig.set_figheight(6)
        #             historyLength = len(aggHistory['loss'])
        #             axL.plot(range(historyLength), np.array(aggHistory['loss']).reshape(-1,1), linestyle='-.', label='training')
        #             axL.plot(range(historyLength), np.array(aggHistory['val_loss']).reshape(-1,1), linestyle='--', label='validation')
        #             axL.set(xlabel='epoch', ylabel='loss')
        #             axA.plot(range(historyLength), np.array(aggHistory['acc']).reshape(-1,1), linestyle='-.', label='training')
        #             axA.plot(range(historyLength), np.array(aggHistory['val_acc']).reshape(-1,1), linestyle='--', label='validation')
        #             axA.set(xlabel='epoch', ylabel='accuracy')
        #             plt.legend()
        #             plt.tight_layout()
        #             fig.subplots_adjust(top=.9)
        #             plt.suptitle('Performance throughout learning')
        #             plt.savefig(os.path.join(modelPath,'perf_in_training_tb_'+str(trainingBatch)+'.png'))
        #             plt.show()
        #
        #             fig, ax1 = plt.subplots(nrows=1, ncols=1)
        #             fig.suptitle('Performance throughout learning')
        #             ax2 = ax1.twinx()
        #             ax1.plot(range(historyLength), np.array(aggHistory['loss']).reshape(-1,1), marker='o', linestyle='-.', label='training')
        #             ax1.plot(range(historyLength), np.array(aggHistory['val_loss']).reshape(-1,1), marker='o', linestyle='--', label='validation')
        #             ax2.plot(range(historyLength), np.array(aggHistory['acc']).reshape(-1,1), marker='.', linestyle='-.', label='training')
        #             ax2.plot(range(historyLength), np.array(aggHistory['val_acc']).reshape(-1,1), marker='.', linestyle='--', label='validation')
        #             plt.legend()
        #             plt.savefig(os.path.join(modelPath,'single_perf_in_training_tb_'+str(trainingBatch)+'.png'))
        #             plt.show()
        #
        #         # restore learning rate corresponding to best model
        #         # sgd.lr.assign(curHist.history['lr'][np.argmin(curHist.history['val_loss'])])
        #
        #         if (verbosity):
        #             bestLossId = np.argmin(curHist.history['val_loss'])
        #             print('final validation loss: %f' % (curHist.history['val_loss'][bestLossId]))
        #             if ('val_acc' in list(curHist.history.keys())):
        #                 print('final validation acc: %f' % ((curHist.history['val_acc'][bestLossId])))
        #             else:
        #                 print('final validation acc: %f' % (curHist.history['val_accuracy'][bestLossId]))
        #
        if (verbosity):
            print('testing commencing')


        if (plotLearning):
            fig, (axL, axA) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
            # fig.set_figwidth(16)
            # fig.set_figheight(6)
            historyLength = len(aggHistory['loss'])
            axL.plot(range(historyLength), np.array(aggHistory['loss']).reshape(-1,1), linestyle='-.', label='training')
            axL.plot(range(historyLength), np.array(aggHistory['val_loss']).reshape(-1,1), linestyle='--', label='validation')
            axL.set(xlabel='epoch', ylabel='loss')
            axA.plot(range(historyLength), np.array(aggHistory['acc']).reshape(-1,1), linestyle='-.', label='training')
            axA.plot(range(historyLength), np.array(aggHistory['val_acc']).reshape(-1,1), linestyle='--', label='validation')
            axA.set(xlabel='epoch', ylabel='accuracy')
            plt.legend()
            plt.tight_layout()
            fig.subplots_adjust(top=.9)
            plt.suptitle('Performance throughout learning')
            plt.savefig(os.path.join(modelPath,'perf_in_training_tb_'+str(trainingBatch+1)+'.png'))
            plt.show()

            fig, ax1 = plt.subplots(nrows=1, ncols=1)
            fig.suptitle('Performance throughout learning')
            ax2 = ax1.twinx()
            ax1.plot(range(historyLength), np.array(aggHistory['loss']).reshape(-1,1), marker='o', linestyle='-.', label='training')
            ax1.plot(range(historyLength), np.array(aggHistory['val_loss']).reshape(-1,1), marker='o', linestyle='--', label='validation')
            ax2.plot(range(historyLength), np.array(aggHistory['acc']).reshape(-1,1), marker='.', linestyle='-.', label='training')
            ax2.plot(range(historyLength), np.array(aggHistory['val_acc']).reshape(-1,1), marker='.', linestyle='--', label='validation')
            plt.legend()
            plt.savefig(os.path.join(modelPath,'single_perf_in_training_tb_'+str(trainingBatch+1)+'.png'))
            plt.show()

        testingBatch = 1

        #test depends on testing paradigm
        if (organParadigm == 'together'):
            curTestingIndices = tm.getIndicesFromAnimalIDs([allAnimalIDs[testingIDs]], imageInterval=imageInterval)
            # curTestingIndices = curTestingIndices[:5]
            # summon the data and labels
            if (verbosity):
                print('summoning data in batch %i' % testingBatch)
                testingBatch += 1
            testingX, testingY = tm.dataSummoner(curTestingIndices, gaussianFilterSigma=gaussianFilterSigma,
                                                 modelType=modelType,
                                                 preprocessingFilterDimensionality=preprocessingFilterDimensionality,
                                                 numberOfChannels=channelCount, preLayerType=preLayerType,
                                                 standardize=standardize, cannySigma=edgeDetection,
                                                 stackedSlices=stackedSlices, numberOfSequences=1 if (usingLSTM or usingConv3D or usingResnet) else None,
                                                 slicesPerSequence = (2*intervalSize if organParadigm=='together' else intervalSize) if (usingLSTM or usingConv3D or usingResnet) else None)

            # cross validate performance
            predictionProbs = ourModel.predict(testingX, batch_size=kBatchSize)
            if (usingLSTM or usingConv3D):
                predictionProbs = predictionProbs[0,:,:]
            predictionDecisions = (predictionProbs > 0.5).astype(int)

            # collect ground truth
            curAnimalY = testingY[0]
            # groundTruth.append(curAnimalY)

            # process predictions
            # if we are using the positiveSlicesThreshold to decide if the animal is positive
            if (positiveSlicesThreshold > -1):
                print("please don't use positivewSlicesThreshold anymore, save that for post analysis and do the naive one in real time")
                # numberOfPositiveSlices = np.sum(predictionDecisions)
                # positivePredictionProbs = np.mean(predictionProbs[np.where(predictionDecisions == 1), :], axis=1)
                # decision = np.reshape(positivePredictionProbs,
                #                       (1, 2)) if numberOfPositiveSlices >= positiveSlicesThreshold \
                #     else np.reshape(np.array((1, 0)), (1, 2))

            # if we are using the naive method (i.e mean across all slices classified(
            else:
                # average confidences
                animalPredictionProbs = np.mean(predictionProbs, axis=0)
                decision = animalPredictionProbs > 0.5
                decision = decision.astype(int)
            # store the decision
            testPredictions = np.concatenate((testPredictions, decision), axis=0)

            # store intermediate results in case we run out of time or get kicked
            # open the shelf
            if (not animalIDList):
                animalIDList = [testingIDs]
                predictionProbabilityList = [predictionProbs]
                groundTruth = [curAnimalY] if not (usingLSTM or usingConv3D) else [curAnimalY[0]]
            else:
                animalIDList.append(testingIDs)
                predictionProbabilityList.append(predictionProbs)
                groundTruth.append(curAnimalY if not (usingLSTM or usingConv3D) else curAnimalY[0])

            ourShelf = shelve.open(shelfName, protocol=0)
            ourShelf['animalIDs'] = animalIDList
            ourShelf['predictionProbabilities'] = predictionProbabilityList
            ourShelf['groundTruth'] = groundTruth
            ourShelf['testPredictions'] = testPredictions
            ourShelf['foldIndex'] = foldIndex
            ourShelf.close()

            # report current results
            if (verbosity):
                print('prediction:%s, truth:%s' % (str(predictionProbs), str(curAnimalY)))
                print('prediction:%s, truth:%s' % (str(decision), str(curAnimalY)))
                
                plot_confusion_matrix(groundTruth, testPredictions.astype(int),
                                      ['normal', 'cancerous'],
                                      normalize=False)

            # free up some memory
            del testingX

        foldIndex += 1
        del ourModel
        del sgd
        tf.keras.backend.clear_session()

        ourModel, sgd = assembleModel(chosenActivation)
        ourModel.set_weights(copy.deepcopy(initialWeights))

    #collect results
    groundTruth = np.array(groundTruth).T.astype(int)
    if (len(np.unique(groundTruth))>1):
        auc = sklearn.metrics.roc_auc_score(groundTruth, testPredictions)
    else:
        auc=-1

    #plot ROC curve
    plt.figure()
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(groundTruth, testPredictions)
    #aggregate these results
    aggregateROCs.append((fpr, tpr, thresholds, auc))

    #thanks to sklearn tutorials for the nicely-formatting boilerplate code
    fig = plt.plot(fpr, tpr, color='red', label='ROC curve (area = %0.2f)'%auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(''.join(['_auc_%s_ROC'%str(auc), modelDescriptorString, '.png']))
    plt.savefig(os.path.join(modelPath, 'roc.png'))

    print(fpr)
    print(tpr)
    print(auc)
    # make some confusion matrices
    fig = plt.figure()
    print(groundTruth)
    print(testPredictions)
    # thanks to sklearn tutorials for prettier confusion matrices
    fig, axis = plot_confusion_matrix(groundTruth, testPredictions.astype(int),
                                      ['normal', 'cancerous'],
                                      normalize=False,
                                      title=''.join(
                                          ['Cancer detector confusion\n',
                                           'tested on ', str(numberOfFolds),
                                           ' folds of leave-one-out cross validation']))

    plt.savefig(''.join(['auc_%s_'%(str(auc)), '_confusions.png']))
    fig.savefig(os.path.join(modelPath, 'cancer_detector_confusions.png'))
    print('model id:')
    print(modelName)
    print(arguments)
    ########################################################################################################################


########################################################################################################################
#compute aggregate ROC curves by interpolating curves and plotting mean +/- 1 std deviation
#iterate over rocs
meanFPR = np.linspace(0, 1, 100)
interpolatedTPRs, aucs = [], []
for curRep in range(numberOfReplications):
    curFPR, curTPR, curThresholds, curAUC = aggregateROCs[curRep]
    interpolatedTPR = np.interp(meanFPR, curFPR, curTPR)
    interpolatedTPR[0] = 0.0
    interpolatedTPRs.append(interpolatedTPR)
    aucs.append(curAUC)

#select the ROC with the best AUC and plot it with -. in black
bestAUC = np.argmax(aucs)
bestFPR, bestTPR = aggregateROCs[bestAUC][0], aggregateROCs[bestAUC][1]

fig, ax = plt.subplots()
#plot result of chance
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
#plot roc with largest area under
ax.plot(bestFPR, bestTPR, linestyle='-.', lw=2, color='k',
        label='AUC = %s'%str(aucs[bestAUC]))
meanTPR = np.mean(interpolatedTPRs, axis=0)
meanTPR[-1] = 1.0
meanAUC = np.mean(aucs)
stdAUC = np.std(aucs)

ax.plot(meanFPR, meanTPR, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (meanAUC, stdAUC),
        lw=2, alpha=.8)

stdTPR = np.std(interpolatedTPRs, axis=0)
tprsUpper = np.minimum(meanTPR + stdTPR, 1)
tprsLower = np.maximum(meanTPR - stdTPR, 0)
ax.fill_between(meanFPR, tprsLower, tprsUpper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver Operating Characteristic")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
ax.legend(loc="lower right")
plt.savefig(modelDescriptorString+'_aggregateROCs.png')
if (not os.path.isdir(pathToIntermediateData)):
    try:
        os.mkdir(basePath)
    except:
        print("could not create path: %s"%str(basePath))
fig.savefig(basePath+'aggregateROCs.png')
if (usingGraphics):
    plt.show()
########################################################################################################################