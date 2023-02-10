########################################################################################################################
# class: tomography manager
# Author: David Schwartz
# 5/21/2019
#
# done:  now I need to write the methods to fetch batches/subbatches of a certain number of samples, label, etc;
#  done: count normal and cancerous images;
#  also done: read in only images between specified indices

# idea: for visualizing data; write a routine to read in histology results and plot histograms of number of cancerous cells per slice index
# currently working on: setting this up as a database tool that dynamically reads images into memory as they're needed;
#
# to traverse the directory structure and collect labeling information from the directory names to
# read images in from files to dataframes or numpy arrays
# consider using python-bioformats instead of PIL
# This class manages collection and organization of tomography imaging data stored on the hard drive
# File organization: "specifiedPathToTopDirectory/machine-learning-OCT/" contains files named according to the
# following convention (which encodes class label/experimental conditions)
# 4/8 - four or eight weeks of age
# W/T - wild type (control) or TAg (cancerous) mice
# V/S - dosed with VCD (menopause) or sesame oil (control)
# for now, we only care about W vs T
#
# in rawTomographyData, we store triples consisting of
# (list of locations of images (type: list of path), uniqueImageIndex (type list of int), label (type array of np.bool))
# and are stored in a dictionary indexed by experimental conditions
# each element of these is a dictionary indexed by  animal identifiers
# each of these is a dictionary indexed by side (i.e. Left or Right represented as 'L' or 'R')
# each of these is a list of Path objects
#
# we also have a new dataFrame, dataByIndex indexed by uniqueImageIndex that stores the above pairs of (image location_list, label_array)
########################################################################################################################


########################################################################################################################
import os
import time
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import imageio
import cv2
from PIL import Image
import scipy
import scipy.ndimage
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from skimage import io
from skimage import filters
from skimage import feature as skfeature
import shutil
import sys
from tensorflow.python.lib.io.file_io import copy
import tqdm
import matplotlib
if sys.platform == 'darwin':
    matplotlib.use("tkAgg")
    print(matplotlib.get_backend())
    import matplotlib.pyplot as plt
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
if 'linux' in sys.platform:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
########################################################################################################################

########################################################################################################################
class tomographyManager:


    #constructor implementation
    def __init__(self, pathToOCTData, verbosity=True):
        imageCount = 0
        #init time
        startTime = time.time()

        #initialize the main data fields
        self.tomographyData, self.dataByLocation = dict(), \
                                                   pd.DataFrame(columns=['label', 'side',
                                                                         'condition', 'imageLocation', 'sliceID'],)
        self.slicesPerAnimal, self.imageRows, self.imageColumns, self.imageCount = 0, 0, 0, 0
        self.normalCount, self.cancerousCount = 0, 0

        #variables to keep track of in case we are standardizing in preprocessing
        self.mean, self.std = None, None

        #see if directory pointed to by pathToOCTData exists
        if (not os.path.exists(pathToOCTData)):
            print('The path specified to the OCTData does not exist!')
            exit(-1)

        else:
            #if the path does exist, look for the data
            p = Path(pathToOCTData)

            #set the image size information
            self.setImageSizeInformation(pathToOCTData)

            #iterate over subdirectories
            subdirectories = [dir for dir in p.iterdir() if dir.is_dir()]
            for curSubDir in subdirectories:
                #parse label from subdirectory name
                curSubDirString = str(curSubDir.name)
                age=curSubDirString[-3]#get the age of the current directory
                label=curSubDirString[-2]#get the label
                curNumericLabel = 1 if label == 'T' else 0
                vcd=curSubDirString[-1]#get the treatment condition
                conditionString = ''.join([age, label, vcd])

                #print the current directorie's info
                if (verbosity):
                    print("age:%s, label:%s, vcd:%s"%(age, label, vcd))

                #formulate our list of animal identification numbers present in the current directory
                animalSubDirs = [curAnimalSubDir for curAnimalSubDir in curSubDir.iterdir()
                                 if curAnimalSubDir.is_dir()]
                uniqueAnimalIdentifiers = [str(curAnimalSubDir.name)[:-2] for curAnimalSubDir in animalSubDirs]
                uniqueAnimalIdentifiers = set(uniqueAnimalIdentifiers) #prune non-unique entries

                #instantiate a dictionary to store this experimental conditions' animals' data
                currentConditionData = dict()
                #iterate over each identifier, checking left, then right directories and place images in a dataframe with
                #age, label, vcd, L/R

                for curID in uniqueAnimalIdentifiers:
                    leftDirPath = Path(os.path.join(pathToOCTData, curSubDirString, ''.join([curID, '_L'])))
                    rightDirPath = Path(os.path.join(pathToOCTData, curSubDirString, ''.join([curID, '_R'])))

                    #list the files in each side
                    leftSubDirList = [dir for dir in leftDirPath.iterdir()]
                    rightSubDirList = [dir for dir in rightDirPath.iterdir()]

                    #count the images
                    imageCountL, imageCountR = len(leftSubDirList), len(rightSubDirList)

                    #read this animal's images into a dictionary
                    curAnimalImages = dict()

                    #sort the image lists
                    leftSubDirList.sort(), rightSubDirList.sort()

                    #compute the labels
                    leftLabels, rightLabels = curNumericLabel*np.ones(shape=(imageCountL,)), \
                                              curNumericLabel*np.ones(shape=(imageCountR,))

                    #compute unique indices
                    largestIndex = imageCount-1
                    uniqueLeftIndices = np.arange(largestIndex, largestIndex+imageCountL)
                    uniqueRightIndices = np.arange(largestIndex+imageCountL-1, largestIndex+imageCountL+imageCountR)

                    #store the locations of the images
                    curAnimalImages['L'] = (leftSubDirList, uniqueLeftIndices, leftLabels)
                    curAnimalImages['R'] = (rightSubDirList, uniqueRightIndices, rightLabels)

                    currentConditionData[curID] = curAnimalImages
                    self.dataByLocation = pd.concat([self.dataByLocation ,
                                                     pd.DataFrame({'label': leftLabels,\
                                                                   'animalID': int(curID),\
                                                                   'side' : 'L',\
                                                                   'age' : int(age),\
                                                                   'condition' : conditionString,\
                                                                   'imageLocation': leftSubDirList,\
                                                                   'sliceID': range(len(leftSubDirList))})],
                                                    axis=0)
                    self.dataByLocation = pd.concat([self.dataByLocation,
                                                     pd.DataFrame({'label': rightLabels, \
                                                                   'animalID': int(curID), \
                                                                   'side' : 'R', \
                                                                   'age' : int(age),\
                                                                   'condition': conditionString, \
                                                                   'imageLocation': rightSubDirList,\
                                                                   'sliceID': range(len(rightSubDirList))})],
                                                    axis=0)

                    #update the number of images stored in our database
                    imageCount += (imageCountL+imageCountR)
                    if (curNumericLabel == 1):
                        self.cancerousCount += (imageCountL+imageCountR)
                    else:
                        self.normalCount += (imageCountL+imageCountR)

                #store the original database
                self.tomographyData[conditionString] = currentConditionData
                self.imageCount = imageCount

        # #set up data augmentation
        # self.generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
        #     samplewise_center=False,  # set each sample mean to 0
        #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #     samplewise_std_normalization=False,  # divide each input by its std
        #     zca_whitening=False,  # apply ZCA whitening
        #     rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        #     horizontal_flip=True,  # randomly flip images
        #     vertical_flip=False)

        #if verbose
        print(''.join(['%i image metadatapoints read into memory in %s seconds'%(self.imageCount, str(time.time()-startTime))]))
        print('data indexing complete')

    #method to get set image size information
    def setImageSizeInformation(self, pathToOCTData='/'):
        #generate a path to the root
        p = Path(pathToOCTData)

        # iterate over subdirectories
        subdirectories = [dir for dir in p.iterdir() if dir.is_dir()]
        for curSubDir in subdirectories:
            # parse label from subdirectory name
            curSubDirString = str(curSubDir.name)
            age = curSubDirString[-3]  # get the age of the current directory
            label = curSubDirString[-2]  # get the label
            vcd = curSubDirString[-1]  # get the treatment condition
            conditionString = ''.join([age, label, vcd])

            # formulate our list of animal identification numbers present in the current directory
            animalSubDirs = [curAnimalSubDir for curAnimalSubDir in curSubDir.iterdir()
                             if curAnimalSubDir.is_dir()]
            uniqueAnimalIdentifiers = [str(curAnimalSubDir.name)[:-2] for curAnimalSubDir in animalSubDirs]
            uniqueAnimalIdentifiers = set(uniqueAnimalIdentifiers)  # prune non-unique entries

            # instantiate a dictionary to store this experimental conditions' animals' data
            currentConditionData = dict()
            # iterate over each identifier, checking left, then right directories and place images in a dataframe with
            # age, label, vcd, L/R
            for curID in uniqueAnimalIdentifiers:
                leftDirPath = Path(os.path.join(pathToOCTData, curSubDirString, ''.join([curID, '_L'])))
                rightDirPath = Path(os.path.join(pathToOCTData, curSubDirString, ''.join([curID, '_R'])))

                # list the files in each side
                leftSubDirList = [dir for dir in leftDirPath.iterdir()]
                rightSubDirList = [dir for dir in rightDirPath.iterdir()]

                #count number of slices we expect
                self.slicesPerAnimal = len(leftSubDirList)

                # get some image size size information
                firstImage = imageio.imread(next(iter(leftSubDirList), None))
                firstImageExample = np.array(firstImage)
                if (firstImageExample is not None):
                    #initilize row and column sizes
                    self.imageRows, self.imageColumns = firstImageExample.shape[0], firstImageExample.shape[1]
                    return self.imageRows, self.imageColumns

                else:
                    print('Error, image read at location %s is None' % str(next(iter(leftSubDirList), None)))

    #method to get a balanced index list from a desired dataset size
    def getBalancedIndices(self, desiredDatasetHalfSize, specificSliceWindow=None, shuffle=False):
        #here we collect all indices split by label and belonging only to the window specified
        #if the indices can belong to any slice
        if (specificSliceWindow is None):
            # get indices of cancerous and normal samples
            cancerousIndices = np.where(self.dataByLocation['label'] == 1)
            normalIndices = np.where(self.dataByLocation['label'] == 0)
            maxAvailableCount = self.cancerousCount

        elif (specificSliceWindow[1] <= self.slicesPerAnimal):
            #get cancerous and normal indices
            cancerousIndices = np.where(self.dataByLocation['label'] == 1)
            normalIndices = np.where(self.dataByLocation['label'] == 0)

            #get indices belonging to the specified slice
            requestedSliceIndices = np.where((self.dataByLocation['sliceID'] >= specificSliceWindow[0]) & \
                                             (self.dataByLocation['sliceID'] <= specificSliceWindow[1]))

            cancerousIndices = np.intersect1d(cancerousIndices, requestedSliceIndices)
            normalIndices = np.intersect1d(normalIndices, requestedSliceIndices)
            maxAvailableCount = np.min([len(normalIndices), len(cancerousIndices)])

        #prune the dataset
        if (desiredDatasetHalfSize > maxAvailableCount):
            print('Warning: unable to collate a dataset of the requested %i samples; \
                    only %i are availabe' % (desiredDatasetHalfSize, maxAvailableCount))
            desiredDatasetHalfSize = maxAvailableCount

        #select indices of images to read in
        indicesToReturn = np.concatenate((normalIndices[:desiredDatasetHalfSize],
                                          cancerousIndices[:desiredDatasetHalfSize]), axis=0)

        if (shuffle):
            np.random.shuffle(indicesToReturn)

        return indicesToReturn

    #method to balance a given set of indices, truncating as few samples as possible
    def balanceIndices(self, prebalancedIndices, shuffle=False):
        balancedIndices = np.zeros(1, len(prebalancedIndices))
        positiveIndices = prebalancedIndices[np.where(self.dataByLocation.iloc[prebalancedIndices]['label']==1)]
        negativeIndices = prebalancedIndices[np.where(self.dataByLocation.iloc[prebalancedIndices]['label']==0)]

        numberOfPositive, numberOfNegative = len(positiveIndices), len(negativeIndices)

        #truncate
        if (numberOfPositive > numberOfNegative):
            balancedIndices = np.concatenate((negativeIndices,positiveIndices[:numberOfNegative]), axis=0)
        else:
            balancedIndices = np.concatenate((positiveIndices, negativeIndices[:numberOfPositive]), axis=0)

        #shuffle if it's specified
        if (shuffle):
            np.random.shuffle(balancedIndices)

        return balancedIndices

    #method to get a list of animal ids corresponding to a specified age (or all ages)
    def getAnimalIDsByAge(self, specifiedAge):
        if (specifiedAge == 0):
            return self.getAllAnimalIDs().astype(int)
        else:
            return pd.unique(self.dataByLocation.iloc[np.where(self.dataByLocation['age']==specifiedAge)]['animalID']).astype(int)

    #method to get a list of animal ids corresponding to a specified age and side
    def getAnimalIDsBySide(self, specifiedAge, specifiedSide=None):
        if (specifiedSide is None):
            return self.getAnimalIDsByAge(specifiedAge)
        elif (specifiedAge == 0):
            return pd.unique(self.dataByLocation.iloc[np.where(self.dataByLocation['side'] == specifiedSide)]['animalID']).astype(int)
        else:
            ageIndices = np.where(self.dataByLocation['age'] == specifiedAge)
            sideIndices = np.where(self.dataByLocation['side'] == specifiedSide)
            ageAndSideIndices = np.intersect1d(ageIndices, sideIndices)
            return pd.unique(self.dataByLocation.iloc[ageAndSideIndices]['animalID']).astype(int)

    #method to collect image indices corresponding to slice indices, age, and side
    def getIndicesBySliceIndex(self, sliceIndex, age=None, side=None):
        #get indices by specified age
        if (age in [None, 0]):
            age = 0
        animalIDs = self.getAnimalIDsByAge(age)
        indices = self.getIndicesFromAnimalIDs(animalIDs, imageInterval=[sliceIndex, sliceIndex], side=side)
        return indices

    #method to get indices of all images in a specific slice window
    def getIndicesBySliceWindow(self, specificSliceWindow=None, age=None, side=None):
        #get indices by specified age
        if (age is None):
            age = 0
        animalIDs = self.getAnimalIDsByAge(age)
        indices = self.getIndicesFromAnimalIDs(animalIDs, imageInterval=specificSliceWindow, side=side)
        return indices

    #method to access all images of label 'non-cancerous (i.e. 0, our null hypothesis)'
    #if we cannot collect sufficient samples, print a warning and select all samples
    def getNormalImages(self, desiredDatasetSize=-1, numberOfChannels=1,
                        specificSliceWindow = None, verbosity=False):
        normalImages = None
        if (verbosity):
            print("collecting normal images")

        normalImages = None

        if (specificSliceWindow is None):
            # get indices of normal samples
            normalIndices = np.where(self.dataByLocation['label'] == 0)
            maxAvailableCount = self.cancerousCount

        elif (specificSliceWindow[1] <= self.slicesPerAnimal):
            normalIndices = np.where(self.dataByLocation['label'] == 0)

            requestedSliceIndices = np.where((self.dataByLocation['sliceID'] >= specificSliceWindow[0]) & \
                                             (self.dataByLocation['sliceID'] <= specificSliceWindow[1]))
            normalIndices = np.intersect1d(normalIndices, requestedSliceIndices)
            maxAvailableCount = len(normalIndices)

        # prune the dataset
        if (desiredDatasetSize > maxAvailableCount):
            print('Warning: unable to collate a dataset of the requested %i samples; \
                            only %i are availabe' % (desiredDatasetSize, maxAvailableCount))
            desiredDatasetSize = maxAvailableCount

        # select indices of images to read in
        indicesToReturn = normalIndices[:desiredDatasetSize]

        # read the images
        cancerousImages = self.getImagesFromIndices(indicesToReturn,
                                                    numberOfChannels=numberOfChannels)

        return cancerousImages

    #method to access all images of label 'cancerous' (i.e. 1)
    def getCancerousImages(self, desiredDatasetSize=-1, numberOfChannels=1,
                           specificSliceWindow = None, verbosity=False):
        if (verbosity):
            print("collecting cancerous images")

        cancerousImages = None

        if (specificSliceWindow is None):
            # get indices of normal samples
            cancerousIndices = np.where(self.dataByLocation['label'] == 1)
            maxAvailableCount = self.cancerousCount

        elif (specificSliceWindow[1] <= self.slicesPerAnimal):
            cancerousIndices = np.where(self.dataByLocation['label'] == 1)

            requestedSliceIndices = np.where((self.dataByLocation['sliceID'] >= specificSliceWindow[0]) & \
                                             (self.dataByLocation['sliceID'] <= specificSliceWindow[1]))
            cancerousIndices = np.intersect1d(cancerousIndices, requestedSliceIndices)
            maxAvailableCount = len(cancerousIndices)

        #prune the dataset
        if (desiredDatasetSize > maxAvailableCount):
            print('Warning: unable to collate a dataset of the requested %i samples; \
                    only %i are availabe' % (desiredDatasetSize, maxAvailableCount))
            desiredDatasetSize = maxAvailableCount

        #select indices of images to read in
        indicesToReturn = cancerousIndices[:desiredDatasetSize]

        #read the images
        cancerousImages = self.getImagesFromIndices(indicesToReturn,
                                                    numberOfChannels=numberOfChannels)

        return cancerousImages


    #method to get normal indices
    def getNormalIndices(self):
        normalIndices = np.where(self.dataByLocation['label']==0)
        return normalIndices

    #method to get cancerous indices
    def getCancerousIndices(self):
        cancerousIndices = np.where(self.dataByLocation['label'] == 1)
        return cancerousIndices

    #method to read images by index
    def getImagesFromIndices(self, indicesToFetch, numberOfChannels=1):
        images = None

        # get paths associated with these images
        imageLocationsColumn = list(self.dataByLocation.keys()).index('imageLocation')
        imagePathsToRead = self.dataByLocation.iloc[indicesToFetch, imageLocationsColumn]
        pathsToRead = [os.path.abspath(imagePathsToRead.values[i]) for i in range(len(imagePathsToRead))]
        if (len(imagePathsToRead) > 0):
            # read the images
            imageCollection = io.imread_collection(pathsToRead, plugin='pil')

            # concatenate the images
            images = io.concatenate_images(imageCollection)

            # # if we need to extend our colorless images to k-channel images
            # if (numberOfChannels > 1):
            #     expandedImages = np.zeros(shape=(images.shape[0], images.shape[1], images.shape[2], numberOfChannels))
            #     # iterate over images and expand them into k-channel images
            #     for curImageIndex in range(0, images.shape[0]):
            #         curImage = images[curImageIndex,:,:]
            #         expandedImages[curImageIndex,:,:,:] = np.repeat(curImage.reshape((curImage.shape[0], curImage.shape[1], 1)),
            #                                                         numberOfChannels, axis=2)
            #     images = expandedImages
                # imagesToReturn = vectorizedChannelMultiplier(imagesToReturn, numberOfChannels)

        return images

    #method to collect indices from a set of animalIDs and slice indices
    #now respects order of animalIDs
    def getIndicesFromAnimalIDs(self, animalIDs, imageInterval=None, side=None):
        requestedIndices = []
        if (imageInterval is not None):
            if (side is not None):
                for curID in animalIDs:
                    curIndices = np.where((self.dataByLocation['sliceID'] >= imageInterval[0]) &\
                                                (self.dataByLocation['sliceID'] <= imageInterval[1]) &\
                                                (self.dataByLocation['side'] == side))[0]
                    curIndices = [requestedIndex for requestedIndex in curIndices if
                                     self.dataByLocation.iloc[requestedIndex]['animalID'] == curID]
                    [requestedIndices.append(curIndex) for curIndex in curIndices]
            else:
                for curID in animalIDs:
                    curIndices = np.where((self.dataByLocation['sliceID'] >= imageInterval[0]) &\
                                                (self.dataByLocation['sliceID'] <= imageInterval[1]))[0]
                    curIndices = [requestedIndex for requestedIndex in curIndices if
                                     self.dataByLocation.iloc[requestedIndex]['animalID'] == curID]
                    [requestedIndices.append(curIndex) for curIndex in curIndices]

            return requestedIndices
        else:
            if (side is not None):
                allIndicesFromIDs = np.where(animalIDs in self.dataByLocation['animalID'] &\
                                             self.dataByLocation['side'] == side)
            else:
                allIndicesFromIDs = np.where(animalIDs in self.dataByLocation['animalID'])
            return allIndicesFromIDs

    #method to get labels by index
    def getLabelsByIndices(self, indicesToFetch):
        labelColumn = list(self.dataByLocation.keys()).index('label')
        labelsToReturn = self.dataByLocation.iloc[indicesToFetch, labelColumn].values
        return labelsToReturn

    #method to get labels by animal id
    #this method now respects the order of animalIDs list
    def getLabelsByAnimalID(self, animalIDs):
        labelToReturn = []
        for curID in animalIDs:
            indicesToFetch = self.getIndicesFromAnimalIDs([curID], imageInterval=[0,0], side='L')
            labelColumn = list(self.dataByLocation.keys()).index('label')
            labelToReturn.append(int(self.dataByLocation.iloc[indicesToFetch, labelColumn].values[0]))
        return labelToReturn

    #method to trim the dataset to restrict to an interval of slices (in a separate copy on the hard drive)
    def trimDataset(self, sliceInterval, newDatasetRootDir):
        #get indices in this slice interval
        smallerDatasetIndices = self.getIndicesBySliceWindow(sliceInterval)

        numberOfIndicesToCopy = len(smallerDatasetIndices)
        itNumber = 0
        #iterate over input paths, use imageio to store in the new locations
        for itNumber in tqdm.tqdm(range(numberOfIndicesToCopy)):
            curIndex = smallerDatasetIndices[itNumber]
            # print("current index: %s of %s"%(str(itNumber),numberOfIndicesToCopy),end='\r')
            # sys.stdout.flush()
            curPath = self.dataByLocation.iloc[curIndex]['imageLocation']
            pathComponentToKeep = str(curPath).split("machine_learning_OCT")[1]
            newPath = os.path.join(newDatasetRootDir,"machine_learning_OCT"+pathComponentToKeep)
            os.makedirs(os.path.dirname(newPath), exist_ok=True)
            shutil.copyfile(curPath, newPath)
        return 0

    # #this method arranges a list of lists of flattened images (one per tomography requested)
    # #as a sequence compatible with the keras BiLSTM structures
    # def getSlicesAsSequences(self, animalIDs, sliceInterval=None, sides=None, verbosity=False):
    #     #get slices specified
    #




    #this method visualizes and stores particular animal's tomograph as a gif
    def visualizeTomographySequence(self, animalID, sliceInterval=None, verbosity=False,
                                    gSigma = 0, cannySigma=None, cannyBounds=[100, 200]):
        #check if we already have data
        if (not self.tomographyData):
            if (verbosity):
                print("missing tomographyData; please instantiate the tomographyManager with uncorrupted data")
        else:
            #get the experimental condition
            expCond = self.getExperimentalConditionByID(str(animalID))
            leftIndices, rightIndices = self.getIndicesFromAnimalIDs([animalID], imageInterval=sliceInterval, side='L'), \
                                        self.getIndicesFromAnimalIDs([animalID], imageInterval=sliceInterval, side='R')

            #get this animal's tomography slices
            #if we're using preprocessing
            preprocessedLeftSlices, y = self.dataSummoner(leftIndices, gaussianFilterSigma=gSigma, verbose=False, numberOfChannels=1,
             modelType='cnn', preprocessingFilterDimensionality=None, stackedSlices=False,
             rawData=False, newDims=[256, 256], slicesPerSequence=None, numberOfSequences=None,
             preLayerType=None, cannySigma=cannySigma, cannyBounds=cannyBounds)
            preprocessedRightSlices, y = self.dataSummoner(rightIndices, gaussianFilterSigma=gSigma, verbose=False, numberOfChannels=1,
             modelType='cnn', preprocessingFilterDimensionality=None, stackedSlices=False,
             rawData=False, newDims=[256, 256], slicesPerSequence=None, numberOfSequences=None,
             preLayerType=None, cannySigma=cannySigma, cannyBounds=cannyBounds)

            #then store these as gifs
            imageio.mimsave(''.join([str(animalID), '_', '_(', str(sliceInterval[0]),'_',str(sliceInterval[1]),')', '_', expCond, '_tomography_L', '.gif']), cv2.normalize(preprocessedLeftSlices, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), fps=24)
            imageio.mimsave(''.join([str(animalID), '_', '_(', str(sliceInterval[0]),'_',str(sliceInterval[1]),')', '_', expCond, '_tomography_R', '.gif']), cv2.normalize(preprocessedRightSlices, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), fps=24)

    #animal id pair for this animal
    def getExperimentalConditionByID(self, animalID):
        curExpCond = None
        #if the animalID exists, iterate over each experimental condition to find it
        for curExpCond in self.tomographyData.keys():
            if (animalID in self.tomographyData[curExpCond].keys()):
                return curExpCond

                # make sure animalID exists
        if (curExpCond is None):
            print("animalID %i could not be found in the database" % animalID)
        #if we get here, we didn't find the animal
        return None

    #routine to split indices across batches
    #this will not randomize the indices by default
    def splitIndicesAcrossBatches(self, indicesToBatchify, batchSize, sequenceLength=None, shuffle=True):
        if (len(indicesToBatchify) == 0 or indicesToBatchify is None or batchSize <=0 ):
            return []

        if (sequenceLength is not None and batchSize < sequenceLength):
            print("batch size < sequence length")

        #if the requested batch size is larger than the array of indices to batchify, just return a list of the array
        if (batchSize > len(indicesToBatchify)):
            return [indicesToBatchify]

        #if shuffle is indicated, randomize the order of the indices
        if (shuffle):
            np.random.shuffle(indicesToBatchify)
        batchifiedIndices = []
        # print('this routine will split indices into a list of arrays, each of which is the set of indices for the corresponding batch')
        #calculate some sizes
        numberOfSamples = len(indicesToBatchify)
        numberOfBatches = np.ceil(numberOfSamples/batchSize)
        lastBatchSize = numberOfSamples % batchSize

        #generate the batches
        batchifiedIndices = [indicesToBatchify[curBatchIndex*batchSize:(curBatchIndex+1)*batchSize] for
                             curBatchIndex in range(numberOfBatches.astype(int))]
        # if (lastBatchSize > 0):
        #     batchifiedIndices.append(indicesToBatchify[(int(numberOfBatches-1)*batchSize):])
        return batchifiedIndices

    def getAllAnimalIDs(self):
        return pd.unique(self.dataByLocation['animalID'])

    #function to calculate mean and standard deviation of a given set of sample indices
    def standardFit(self, sampleIndices, cannySigma=None, preLayerType=None, modelType=None):
        #summon data
        unnormalizedData, labels = self.dataSummoner(sampleIndices, gaussianFilterSigma=1, modelType=modelType,
                                             preLayerType=preLayerType, cannySigma=cannySigma, standardize=False)

        #calculate statistics
        self.mean = np.mean(unnormalizedData, axis=(0, 1, 2, 3))
        self.stddev = np.std(unnormalizedData, axis=(0, 1, 2, 3))

        return (self.mean, self.stddev)


    # define a data-organization-and-pre-processing routine to summon only the data we need
    # returns the pair (preprocessedData, y)
    # or a pair of lists (sequences, y)
    #edsc argument controls if data are returned with canny output appended as a second channel
    def dataSummoner(self, sampleIndices, gaussianFilterSigma, verbose=False, numberOfChannels=1,
                     modelType=None, preprocessingFilterDimensionality=None, stackedSlices=False,
                     rawData=False, newDims=None, slicesPerSequence=None, numberOfSequences=None,
                     preLayerType=None, cannySigma=None, cannyBounds=[100, 200],
                     binocular=False, edsc=False, standardize=False):
        # collect images
        rawX = self.getImagesFromIndices(sampleIndices, numberOfChannels=numberOfChannels)

        if (stackedSlices is not None and stackedSlices):
            sampleIndices_plus1 = [index + 1 for index in sampleIndices]
            sampleIndices_minus1 = [index - 1 for index in sampleIndices]

            rawX_plus1 = self.getImagesFromIndices(sampleIndices_plus1, numberOfChannels=1)
            rawX_minus1 = self.getImagesFromIndices(sampleIndices_minus1, numberOfChannels=1)

        # collect the labels
        y = self.getLabelsByIndices(sampleIndices)

        #compress images if arguments indicate
        if (newDims is not None and len(newDims) == 2):
            resizedX = np.zeros(shape=(rawX.shape[0], newDims[0], newDims[1]))
            for i in range(rawX.shape[0]):
                resizedX[i,:,:] = resize(rawX[i], (newDims[0], newDims[1]))
            rawX = resizedX

        #if we want to ignore preprocessing:
        if (rawData):
            return rawX, y

        if (verbose):
            print('data organized')
            print('X has dimensions %s' % str(rawX.shape))
            print('(sample, pixel row, pixel column)')
            print('preprocessing the data')

        # gaussian filtering
        if (preprocessingFilterDimensionality == 3 and gaussianFilterSigma > 0):
            filteredX = scipy.ndimage.gaussian_filter(rawX, sigma=gaussianFilterSigma)

            #if we're using Canny edge detection
            if (cannySigma is not None):
                for curImageIndex in range(rawX.shape[0]):
                    if (cannyBounds == 'ROT'):
                        medianPixelValue = np.median(filteredX.reshape((-1, )))
                        cannyBounds = [0.66*medianPixelValue, 1.33*medianPixelValue]
                    filteredX[curImageIndex, :, :] = cv2.Canny(cv2.normalize(filteredX[curImageIndex, :, :], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cannyBounds[0], cannyBounds[1], cannySigma)


            if (stackedSlices):
                filteredX_plus1 = scipy.ndimage.gaussian_filter(rawX_plus1, sigma=gaussianFilterSigma)
                filteredX_minus1 = scipy.ndimage.gaussian_filter(rawX_minus1, sigma=gaussianFilterSigma)

        else:
            # if not using the 3d gaussian filtering, then check if were using stackedSlices
            # this needs to be changed later, so stackedSlices and 3d filtering arent mutually exclusive
            filteredX = np.zeros(shape=rawX.shape)

            if stackedSlices:
                filteredX_plus1 = np.zeros(shape=rawX.shape)
                filteredX_minus1 = np.zeros(shape=rawX.shape)

            if (gaussianFilterSigma > 0):
                for curImageIndex in range(rawX.shape[0]):
                    filteredX[curImageIndex, :, :] = filters.gaussian(rawX[curImageIndex, :, :], sigma=gaussianFilterSigma, preserve_range=True)

                    if stackedSlices:
                        filteredX_plus1[curImageIndex, :] = filters.gaussian(rawX_plus1[curImageIndex, :, :],
                                                                             sigma=gaussianFilterSigma, preserve_range=True)
                        filteredX_minus1[curImageIndex, :] = filters.gaussian(rawX_minus1[curImageIndex, :, :],
                                                                              sigma=gaussianFilterSigma, preserve_range=True)
            else:
                filteredX = rawX

            #if we're using Canny edge detection
            if (cannySigma is not None):
                for curImageIndex in range(rawX.shape[0]):
                    if (cannyBounds == 'ROT'):
                        medianPixelValue = np.median(filteredX.reshape((-1,)))
                        cannyBounds = [0.66*medianPixelValue, 1.33*medianPixelValue]
                    filteredX[curImageIndex, :, :] = cv2.Canny(cv2.normalize(filteredX[curImageIndex, :, :], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cannyBounds[0], cannyBounds[1], cannySigma)
        if (modelType != 'mlp' and modelType != 'cnn'):
            # expand the images manually here instead of inside the TomographyManager
            # so that we don't have to apply the gaussian filter to 3 copies of the same image

            if stackedSlices:
                expandedImages = np.stack([filteredX_plus1, filteredX, filteredX_minus1], axis=-1)

            else:
                expandedImages = np.zeros(
                    shape=(filteredX.shape[0], filteredX.shape[1], filteredX.shape[2], numberOfChannels))

                # iterate over images and expand them into k-channel images
                for curImageIndex in range(0, filteredX.shape[0]):
                    curImage = filteredX[curImageIndex, :, :]
                    expandedImages[curImageIndex, :, :, :] = np.repeat(
                        curImage.reshape((curImage.shape[0], curImage.shape[1], 1)),
                        numberOfChannels, axis=2)
        else:
            expandedImages = filteredX

        # apply preprocessing depending on model type:
        if (modelType == 'vgg' or (modelType == 'lstm' and preLayerType == 'vgg')):
            # preprocessedX = filteredX/255
            preprocessedX = vgg_preprocess_input(filteredX)
            expandedImages = np.zeros(shape=(filteredX.shape[0], filteredX.shape[1], filteredX.shape[2], numberOfChannels))
            # iterate over images and expand them into k-channel images
            for curImageIndex in range(0, filteredX.shape[0]):
                curImage = preprocessedX[curImageIndex, :, :]
                expandedImages[curImageIndex, :, :, :] = np.repeat(
                    curImage.reshape((curImage.shape[0], curImage.shape[1], 1)),
                    numberOfChannels, axis=2)
            preprocessedX = expandedImages
            # if (self.mean == None or self.stddev == None):
            #     print("since you're using vgg somewhere, you need standardization to be on")
            #     preprocessedX = expandedImages
            # else:
            #     preprocessedX = (expandedImages-self.mean)/(self.stddev + 0.0000001)

        elif (modelType == 'cnn' or modelType == 'babyVGG' or (modelType=='lstm' and preLayerType=='cnn')):
            # scale the input
            preprocessedX = np.expand_dims(filteredX, axis=3)/255.

        elif (modelType == 'vgg19'):
            preprocessedX = vgg19_preprocess_input(expandedImages)

        elif (modelType == 'xception'):
            preprocessedX = xception_preprocess_input(expandedImages)

        # elif (modelType == 'resnet'):
        #     preprocessedX = resnet_preprocess_input(expandedImages)

        # elif (modelType == 'unet'):
        #     unet_preprocess_input = sm.get_preprocessing('resnet34')
        #     preprocessedX = unet_preprocess_input(expandedImages)

        elif (modelType == 'mlp' or preLayerType == 'mlp'):
            # flatten input here so we can scale it all at once
            imageExample = filteredX[0, :]
            imWidth, imHeight = imageExample.shape[0], imageExample.shape[1]

            # flatten and scale the input
            preprocessedX = filteredX.reshape((filteredX.shape[0], imWidth * imHeight))/255.

        elif((modelType == ' ') and preLayerType == 'cnn'):
            imageExample = filteredX[0, :]
            imWidth, imHeight = imageExample.shape[0], imageExample.shape[1]

            #scale the input
            preprocessedX = np.expand_dims(filteredX, axis=3)/255. - 0.5

        elif ((modelType == 'c3d' or modelType == 'resnet') and preLayerType == 'cnn'):
            imageExample = filteredX[0, :]
            imWidth, imHeight = imageExample.shape[0], imageExample.shape[1]

            # scale the input
            preprocessedX = np.expand_dims(filteredX, axis=3)/255.


        #if we're standardizing the data, do that here
        if (standardize):
            preprocessedX = (preprocessedX-self.mean)/(self.stddev + 0.0000001)

        if (verbose):
            print('data preprocessed')
            print('preprocessed X has dimensions %s' % str(preprocessedX.shape))
            print('(sample, pixel row, pixel column, channel)')

        #if we're appending edge detection output as an additional channel
        if (edsc):
            pass
            #this isn't curerently implemented, but eventually, make sure that when this is used, the appended ED images
            #are processed from the filtered images no the scaled images in the earlier steps
            # jointOutput = np.zeros(shape=(filteredX.shape[0], filteredX.shape[1], filteredX.shape[2], 2))
            # jointOutput[:,:,:,0] = preprocessedX
            # for curImageIndex in range(rawX.shape[0]):
            #     if (cannyBounds == 'ROT'):
            #         medianPixelValue = np.median(filteredX.reshape((-1,)))
            #         cannyBounds = [0.66 * medianPixelValue, 1.33 * medianPixelValue]
            #     jointOutput[curImageIndex, :, :, 1] = cv2.Canny(cv2.normalize(filteredX[curImageIndex, :, :],
            #                                                                   None, 0, 255, cv2.NORM_MINMAX,
            #                                                                   cv2.CV_8U),
            #                                                     cannyBounds[0], cannyBounds[1], cannySigma)
        # free up some memory
        del filteredX
        del rawX

        #consider splitting sequences
        if (numberOfSequences is not None and slicesPerSequence is not None):
            #exception handling
            if (numberOfChannels>1):
                print("Only single channel LSTMs are supported for now")
                exit(-1)
            if (slicesPerSequence < 0):
                print("This routine needs to know the number of slices per sequence")
                print("supplied sequences had specified length: %s"%str(len(slicesPerSequence)))
                exit(-1)

            sequences, labels = [], []
            labels = [[y[(i)*(slicesPerSequence)+j] for j in range(slicesPerSequence)] for i in range(numberOfSequences)]
            # labels = np.array(labels).reshape(-1,1)
            sequences = np.array([[preprocessedX[curSequence*(slicesPerSequence)+curImageIndex,:,:,:] for curImageIndex in range(slicesPerSequence)]
                         for curSequence in range(numberOfSequences)])

            return sequences, labels

        else:
            return preprocessedX, y

    #routine that rotates images by a multiple of pi/4

    #routine that reflects images across the vertical line at the center of the image

    #routine that reflects images across the horizontal line at the center of the image

    #routine that reflects images across the diagonal line from bottom left to top right

    #routine that reflects images across the diagonal line from bottom right to top left

    #routine to visualize an organ in 3d
    def visualize3D(self, animalID, side='R'):
        #todo: consider tomViz? or maybe openCV has something?
        return None

########################################################################################################################


########################################################################################################################
#if we're running this as a script, test the class
if __name__ == "__main__":


    #set up argument parser
    parser = argparse.ArgumentParser()

    # python windowSweeper.py -v -g 1 -t pratikdesktop_logical -w 250 -nw 4 -trw 76199170 76433824 -tew 331530177 333795633 -sp hashing -f 120 -pn firefox
    #define the arguments
    parser.add_argument('-v', '--verbose', help='increased console verbosity',
                        action='store_true', required=False)
    parser.add_argument('-g', '--graphical', help='plot graphics as we run',
                        type=bool, required=False)
    parser.add_argument('-p', '--pathToTomographyFiles', help='absolute path to the machine_learning_OCT directory',
                        type=str, required=False)

    #parse the arguments
    arguments = parser.parse_args()
    verbosity = arguments.verbose if arguments.verbose is not None else False #default verbosity set to false
    if (verbosity):
        print(arguments)
    pathToTomographyFiles = arguments.pathToTomographyFiles if arguments.pathToTomographyFiles is not None else 'E:\\bigData\\machine_learning_OCT'#default path is root directory

    #instantiate the tomography data manager
    tm = tomographyManager(pathToTomographyFiles, verbosity)
    print('dataset read')


    #tsne decomposition question for reviewer 2
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt    
    from matplotlib.ticker import NullFormatter
    from sklearn import manifold, datasets
    from time import time
    import itertools
    import copy
    oneSamplePerAnimal = False
    edSigma = None
    red, green = -1, -1
    perplexities = [5, 30, 50, 100]
    sliceInterval = [75, 125]
    intervalLength = sliceInterval[1]-sliceInterval[0]+1
    #summon the data
    # cancerousIndices = tm.getCancerousIndices(specificSliceWindow=sliceInterval, verbosity=True)
    # benignIndices = tm.getNormalIndices(specificSliceWindow=sliceInterval, verbosity=True)
    #select holdout validation IDs
    allAnimalIDs = tm.getAnimalIDsByAge(8)
    IDsY = np.array(tm.getLabelsByAnimalID(allAnimalIDs))
    IDsPos, IDsNeg = allAnimalIDs[np.where(IDsY == 1)[0]],\
                                        allAnimalIDs[np.where(IDsY == 0)[0]]
    indicesPos, indicesNeg = tm.getIndicesFromAnimalIDs(IDsPos.tolist(), imageInterval=sliceInterval), tm.getIndicesFromAnimalIDs(IDsNeg.tolist(), imageInterval=sliceInterval)

    if (not oneSamplePerAnimal):
        cancerousImages, cancerousLabels = tm.dataSummoner(indicesPos, gaussianFilterSigma=1,
                                        modelType='cnn',
                                        preprocessingFilterDimensionality=2,
                                        preLayerType='cnn', standardize=False, 
                                        cannySigma=edSigma, stackedSlices=None)
                                        # numberOfSequences=len(IDsPos),
                                        # slicesPerSequence = 2*intervalLength)
        print('shape of cancerous data')
        print(cancerousImages.shape)
        benignImages, benignLabels = tm.dataSummoner(indicesNeg, gaussianFilterSigma=1,
                                        modelType='cnn',
                                        preprocessingFilterDimensionality=2,
                                        preLayerType='cnn', standardize=False, 
                                        cannySigma=edSigma, stackedSlices=None)
                                        # numberOfSequences=len(IDsPos),
                                        # slicesPerSequence = 2*intervalLength)
        print('shape of benign data')
        print(benignImages.shape)    
        #organize the data into a single labelled array
        concatenatedDataset = np.concatenate((cancerousImages.squeeze(), benignImages.squeeze()), axis=0)
        concatenatedDataset = concatenatedDataset.reshape((concatenatedDataset.shape[0], -1))
        datasetLabels = np.concatenate((cancerousLabels, benignLabels),axis=0)
    
    else:
        cancerousImages, cancerousLabels = tm.dataSummoner(indicesPos, gaussianFilterSigma=1,
                                modelType='cnn',
                                preprocessingFilterDimensionality=2,
                                preLayerType='cnn', standardize=False, 
                                cannySigma=edSigma, stackedSlices=None,
                                numberOfSequences=len(IDsPos),
                                slicesPerSequence = 2*intervalLength)
        print('shape of cancerous data')
        print(cancerousImages.shape)
        benignImages, benignLabels = tm.dataSummoner(indicesNeg, gaussianFilterSigma=1,
                                        modelType='cnn',
                                        preprocessingFilterDimensionality=2,
                                        preLayerType='cnn', standardize=False, 
                                        cannySigma=edSigma, stackedSlices=None,
                                        numberOfSequences=len(IDsNeg),
                                        slicesPerSequence = 2*intervalLength)
        print('shape of benign data')
        print(benignImages.shape)    
        #organize the data into a single labelled array
        concatenatedDataset = np.concatenate((cancerousImages.squeeze(), benignImages.squeeze()), axis=0)
        concatenatedDataset = concatenatedDataset.reshape((concatenatedDataset.shape[0], -1))
        datasetLabels = np.concatenate((np.ones((len(IDsPos),1)), np.zeros((len(IDsNeg),1))), axis=0)#np.concatenate((cancerousLabels, benignLabels),axis=0)
    red = np.isclose(datasetLabels, 1).reshape(-1,)
    green = np.isclose(datasetLabels,0).reshape(-1,)
    #calculate TSNE decomposition in 2 dims
    (fig, subplots) = plt.subplots(len(perplexities), figsize=(15, 15))

    for i, perplexity in enumerate(perplexities):
        ax = subplots[i]

        t0 = time()
        embed2D = TSNE(n_components=2, perplexity=perplexity)
        Y = embed2D.fit_transform(concatenatedDataset)
        t1 = time()
        print("OCT embedding perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
        #plot 2dim embedding
        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[red, 0], Y[red, 1], c="r", label='positive')
        ax.scatter(Y[green, 0], Y[green, 1], c="g", label='negative')
        # ax.axis("tight")
    plt.legend()
    plt.show()
    plt.savefig('2dEmbedding.pdf')




    #shrink the dataset
    # print("shrinking the dataset to between slices 50 and 250")
    # result = tm.trimDataset([50, 300], "C:\\bigData\\ovarianCancerDetection\\_50_250_OCT")
    # #test the batch splitting routine
    # a = np.arange(100)
    # b = np.arange(103)
    # c = tm.splitIndicesAcrossBatches(a, 10)
    # d = tm.splitIndicesAcrossBatches(b, 5)
    # print(a)
    # print(c)
    # print(b)
    # print(d)
    # # #test visualization of animal 3767/3765
    #
    # gSigma=1
    # cannySigma=2
    # tm.visualizeTomographySequence(3767, gSigma=gSigma,
    #                                         sliceInterval=[100, 200])
    # tm.visualizeTomographySequence(3765, gSigma=gSigma,
    #                                         sliceInterval=[100, 200])


    #2d gaussian plotter for conv illustration
    #mlab function:
    def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                         mux=0.0, muy=0.0, sigmaxy=0.0):
        """
        Bivariate Gaussian distribution for equal shape *X*, *Y*.

        See `bivariate normal
        <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
        at mathworld.
        """
        Xmu = X - mux
        Ymu = Y - muy

        rho = sigmaxy / (sigmax * sigmay)
        z = Xmu ** 2 / sigmax ** 2 + Ymu ** 2 / sigmay ** 2 - 2 * rho * Xmu * Ymu / (sigmax * sigmay)
        denom = 2 * np.pi * sigmax * sigmay * np.sqrt(1 - rho ** 2)
        return np.exp(-z / (2 * (1 - rho ** 2))) / denom
    # Mean vector and covariance matrix
    #set up the input data
    # mu = np.array([0,0])
    # Sigma = np.array([[1.0, 0], [0, 1.0]])
    # x = np.arange(-4, 4, 0.0125)
    # y = np.arange(-4, 4, 0.0125)
    # x,y = np.meshgrid(x,y)
    # z = bivariate_normal(x, y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0)

    # #make the figure
    # plt.figure()
    # ourplot = plt.contourf(x,y,z)
    # plt.savefig('gaussianPlot')
    # plt.show()
    # tm.visualizeTomographySequence(3767, gSigma=gSigma, cannySigma=cannySigma,
    #                                         cannyBounds=[100, 200],
    #                                         sliceInterval=[100, 200])
    # tm.visualizeTomographySequence(3505, gSigma=1.0, cannySigma=1.0,
    #                                         sliceInterval=[100, 200])
    # tm.visualizeTomographySequence('3328', sliceInterval=[200, 3000])

    # #plot clipping elu activation function
    # import tensorflow
    # def clipping_relu(x):
    #     return tensorflow.clip_by_value(tensorflow.nn.elu(x),
    #                                     tensorflow.constant(-1.0),
    #                                     tensorflow.constant(1.0))
    #
    # xVals = np.linspace(-2.4, 1.4, 100)
    # yValsCELU = tensorflow.keras.backend.eval(clipping_relu(tf.keras.backend.variable(xVals)))
    # fig, ax = plt.subplots()
    # ourplot = ax.plot(xVals, yValsCELU, label='clipped exponential linear')
    # plt.xlabel('pre-activation domain')
    # plt.ylabel('activation range')
    # plt.title('clipped exponential linear activation')
    # ax.set_aspect('equal')
    # ax.grid(True, which='both')
    # plt.savefig('elu_plot.png')
    # xVals = np.linspace(-5, 5, 1000)
    # yValsCELU = tensorflow.keras.backend.eval(clipping_relu(tf.keras.backend.variable(xVals)))
    # yValsSoftMax = tensorflow.keras.backend.eval(tf.nn.softmax(tf.keras.backend.variable(xVals)))
    # fig, ax = plt.subplots()
    # ourplot = ax.plot(xVals, yValsCELU, label='clipped exponential linear activation')
    # ourplot = ax.plot(xVals, yValsSoftMax, label='soft-max activation')
    # plt.xlabel('pre-activation domain')
    # plt.ylabel('activation range')
    # plt.title('activation functions')
    # plt.legend()
    # ax.set_aspect('equal')
    # ax.grid(True, which='both')
    # plt.savefig('elu_plot.png')


########################################################################################################################