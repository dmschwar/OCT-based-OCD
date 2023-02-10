########################################################################################################################
#Author: David Schwartz
#
# This script takes an argument to a directory containing shelves from an earlier run of the loo experiment
# and produces an aggregate ROC figure with confidence intervals as in
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
########################################################################################################################
########################################################################################################################
#imports
import os, sys
import numpy as np
import shelve
import sklearn
import argparse
from sklearn import *
# import tensorflow as tf
import matplotlib
#control matplotlib backend
if sys.platform == 'darwin':
    matplotlib.use("tkAgg")
    print(matplotlib.get_backend())
    import matplotlib.pyplot as plt
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
if 'linux' in sys.platform:
    usingLinux = True
else:
    usingLinux = False
import matplotlib.pylab as plt
########################################################################################################################

########################################################################################################################
#set up argument parser
parser = argparse.ArgumentParser()

#define the arguments
parser.add_argument('-p', '--pathToIntermediateData', help='absolute path to the intermediate data directory where shelves are stored',
                    type=str, required=True)
parser.add_argument('-bp', '--basePath', help='working directory where we want the plot produced by this script to be stored',
                    type=str, required=True)
parser.add_argument('-md', '--modelDescriptor', help='the numerical ID of the experiment we wish to plot results of',
                    type=str, required=True)

#parse the arguments
arguments = parser.parse_args()

#set up arguments (just a directory where shelves are stored)
pathToIntermediateData = arguments.pathToIntermediateData if arguments.pathToIntermediateData is not None else '/intermediateData/OCD/'
basePath = arguments.basePath if arguments.basePath is not None else '/Users/David/Dropbox/David-Greg/Projects/ovarianCancerDetection/ovarian-cancer-detection/'
modelDescriptorString = arguments.modelDescriptor if arguments.modelDescriptor is not None else '1596698346024'

#read in shelves, load their data if possible, if not, print a warning and skip the current shelf
#in doing this, aggregate curFPR, curTPR, curThresholds, curAUC from the stored data
#shelf files are of the form "cd_shelf<modelIndentifierString>_index.shelf
#so let's list all files and grab those containing "shelf<modelDescriptorString>"
#
if (not usingLinux):
    fileList = [f for f in os.listdir(pathToIntermediateData) if (modelDescriptorString in str(f) and
                                                                os.path.isfile(os.path.join(pathToIntermediateData, f)) and
                                                                '.dat' in str(f)
                                                                )]
else:
    fileList = [f for f in os.listdir(pathToIntermediateData) if (modelDescriptorString in str(f) and
                                                                  os.path.isfile(
                                                                      os.path.join(pathToIntermediateData, f)) and
                                                                  '.db' in str(f)
                                                                  )]
print(fileList)
#aggregate the ROCs by reading
#iterate over shelves:
#collect results
aggregateROCs = []
for curFile in fileList:
    shelfFile = curFile[:-4] if not usingLinux else curFile[:-3]
    #read in variables from the shelf
    ourShelf = shelve.open(os.path.join(pathToIntermediateData, shelfFile), protocol=0)
    groundTruth, testPredictions = ourShelf['groundTruth'], ourShelf['testPredictions']

    groundTruth = np.array(groundTruth).T.astype(int)
    if (len(np.unique(groundTruth))>1):
        auc = sklearn.metrics.roc_auc_score(groundTruth, testPredictions[:,1])
    else:
        auc=-1

    #plot ROC curve
    plt.figure()
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(groundTruth, testPredictions[:,1])
    #aggregate these results
    aggregateROCs.append((fpr, tpr, thresholds, auc))

numberOfReplications = len(aggregateROCs)

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
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
#plot roc with largest area under
ax.plot(bestFPR, bestTPR, linestyle='-.', lw=2, color='k',
        label='AUC = %0.2g'%aucs[bestAUC])

meanTPR = np.mean(interpolatedTPRs, axis=0).astype(np.float)
meanAUC = np.mean(aucs)
stdAUC = np.std(aucs)/np.sqrt(numberOfReplications)

ax.plot(meanFPR, meanTPR, color='b',
            label=r'Mean ROC (AUC = %0.2g $\pm$ %0.2g)' % (meanAUC, stdAUC),
        lw=2, alpha=.8)

stdTPR = np.std(interpolatedTPRs, axis=0)/np.sqrt(numberOfReplications)
tprsUpper = np.minimum(meanTPR + stdTPR, 1)
tprsLower = np.maximum(meanTPR - stdTPR, 0)
ax.fill_between(meanFPR, tprsLower, tprsUpper, color='grey', alpha=.2,
                label=r'$\pm$ S.E.M.')

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
fig.savefig(os.path.join(basePath, '_aggregateROCs.png'))
plt.show()

########################################################################################################################

