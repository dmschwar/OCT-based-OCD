# manualThrsholds.py
# This script was primarily authored by Noah Thurston.
# In 03/2020 this is being adapted to
#
# This script is to test to see if static thresholds can aid in classification of a sequence of slice labels.
# It works by loading the cancer/not-cancer model outputs from a .shelf, and then interpolating over a sequence of
# thresholds to see how many misclassifications each boundary creates.

import os
import sys
import shelve
import numpy as np
import matplotlib
import sklearn

from sklearn import metrics
import time
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

import matplotlib.pylab as plt


#special thanks to Jake Walden on StackOverflow for writing a vectorized ewma that's faster than pandas
def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out

def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None, cmap=plt.cm.Blues):
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
    # Only use the labels that appear in the data
    # classes = classes[sklearn.utils.multiclass.unique_labels(y_true, y_pred)]

    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    #
    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
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
    return fig, ax, cm



name = "cancer_detector_shelfmodel_type_vgg_gsigma_20.0_as_8_lf_-1_ii_[100, 150]_epochs_50_patience_25_bs_500_lt_0.01_PS_[512]_alpha_0.314159_dr_0.5_pst_-1_pfd_2.0_ss_False.shelf"
ourShelf = shelve.open(name)
print(list(ourShelf.keys()))
predictionProbs_all = ourShelf['predictionProbabilities']
groundTruth_all = ourShelf['groundTruth']
plt.plot(groundTruth_all)
plt.show()
numberOfFolds = len(groundTruth_all)


#iterate over animals and plot p_y=1, ewma(p_y=1) for varying alphas


window_size = 50
prob_threshold = 0.50
window_threshold_values = list(np.arange(0.05, 0.95, 0.05))
alpha_values = list(np.arange(0.2, 0.35, 0.05))
num_false_pos = []
num_false_neg = []
#
# for window_threshold in window_threshold_values:
#
#
#     print("\nwindow_threshold: {}".format(window_threshold))
#
#     decisions = []
#
#     for index in range(numberOfFolds):
#         predictionProbs = predictionProbs_all[index]
#         predictionProbsRounded = [(1 if prob[1]>prob_threshold else 0) for prob in predictionProbs]
#
#         window_sums = []
#         for start_index in range(0, 100-window_size):
#             window_sums.append(np.sum(predictionProbsRounded[start_index:start_index+window_size]))
#
#
#         max_window_sum = np.max(window_sums)
#
#
#         if max_window_sum >= window_threshold:
#             decisions.append(1)
#         else:
#             decisions.append(0)
#
#
#     fig, axis, cm = plot_confusion_matrix(groundTruth_all, decisions, ['normal', 'cancerous'], normalize=False, title='test title')
#     # plt.show()
#     # print(cm.shape)
#
#     num_false_pos.append(cm[0][1])
#     num_false_neg.append(cm[1][0])
#
#
#
# total_falses = [(false_poss + false_negs ) for (false_poss, false_negs) in zip(num_false_pos, num_false_neg)]
#
# print("num_false_pos: {}".format(num_false_pos))
# print("num_false_neg: {}".format(num_false_neg))
#
# plt.clf()
#
# plt.title("Misclassifications vs Window Sum Threshold (min at ({},{}))".format(window_threshold_values[np.argmin(total_falses)], np.min(total_falses)))
# plt.xlabel("Threshold of the Window Sum")
# plt.ylabel("Number of Misclassifications")
#
# plt.plot(window_threshold_values, num_false_pos, linestyle='-.', label="False Positives")
# plt.plot(window_threshold_values, num_false_neg, linestyle='-.', label="False Negatives")
# plt.plot(window_threshold_values, total_falses, linestyle=':', label="Total False")
#
# plt.legend(loc="upper right")
#
# plt.savefig("window_sweep_probtuned.png")



for curWindowThreshold in window_threshold_values:
    num_false_pos = []
    num_false_neg = []
    num_false_pos_ewma = []
    num_false_neg_ewma = []
    for alpha in alpha_values:


        print("\newma alpha: {}".format(alpha))

        decisions, decisions_ewma = [], []

        for index in range(numberOfFolds):
            predictionProbs = predictionProbs_all[index]
            print(predictionProbs.shape)
            predictionProbs_ewma = ewma_vectorized(predictionProbs[:,1].reshape(-1,), alpha)
            plt.figure()
            # plt.plot(window_threshold*np.ones((len(predictionProbs),)),linestyle='-', color='r')
            # plt.plot(predictionProbs.reshape(-1,))
            # plt.plot(predictionProbs_ewma.reshape(-1,))
            plt.stem(predictionProbs[:,1].reshape(-1,), markerfmt='-o', basefmt=None)
            plt.stem(predictionProbs_ewma, markerfmt='-D', basefmt=None)
            plt.title('ground truth: %s'%("positive" if groundTruth_all[index] > 0 else "negative"))
            plt.legend(['raw', 'ewma'], loc="best")
            plt.tight_layout()
            plt.show()
            predictionProbsRounded = [(1 if prob>prob_threshold else 0) for prob in list(predictionProbs)]

            if np.mean(predictionProbs_ewma) >= curWindowThreshold:
                decisions_ewma.append(1)
            else:
                decisions_ewma.append(0)
            if np.mean(predictionProbs) >= curWindowThreshold:
                decisions.append(1)
            else:
                decisions.append(0)

        fig, axis, cm = plot_confusion_matrix(groundTruth_all, decisions, ['normal', 'cancerous'], normalize=False, title='test title')
        num_false_pos.append(cm[0][1])
        num_false_neg.append(cm[1][0])

        fig, axis, cm = plot_confusion_matrix(groundTruth_all, decisions_ewma, ['normal', 'cancerous'], normalize=False, title='test title')
        num_false_pos_ewma.append(cm[0][1])
        num_false_neg_ewma.append(cm[1][0])

    total_falses = [(false_poss + false_negs ) for (false_poss, false_negs) in zip(num_false_pos, num_false_neg)]
    total_falses_ewma = [(false_poss + false_negs) for (false_poss, false_negs) in zip(num_false_pos_ewma, num_false_neg_ewma)]

    print("num_false_pos: {}".format(num_false_pos))
    print("num_false_neg: {}".format(num_false_neg))

    plt.clf()

    plt.title("Misclassifications vs ewma $\\alpha$ (min at ({},{}))".format(window_threshold_values[np.argmin(total_falses)], np.min(total_falses)))
    plt.xlabel("$\\alpha$")
    plt.ylabel("Number of Misclassifications")

    plt.plot(alpha_values, num_false_pos_ewma, linestyle='-.', label="False Positives (ewma)")
    plt.plot(alpha_values, num_false_neg_ewma, linestyle='-.', label="False Negatives (ewma)")
    plt.plot(alpha_values, total_falses_ewma, linestyle=':', label="Total False (ewma)")

    plt.plot(alpha_values, num_false_pos, linestyle='-.', label="False Positives")
    plt.plot(alpha_values, num_false_neg, linestyle='-.', label="False Negatives")
    plt.plot(alpha_values, total_falses, linestyle=':', label="Total False")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("alpha_sweep_probtuned_wt_%s.png"%str(curWindowThreshold))
