import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_roc(test_label, preds_prob, target_names=None, figsize=None):
    if test_label.shape == preds_prob.shape:
        if preds_prob.ndim == 2 and test_label.ndim == 2:
            num_classes = test_label.shape[1]
        elif preds_prob.ndim == 1 and test_label.ndim == 1:
            num_classes = 1
        else:
            raise RuntimeError('Invalid dimentions')
    else:
        raise RuntimeError('Invalid shapes')

    if figsize:
        plt.figure(figsize=figsize)
    for cl in range(num_classes):
        # calculate the fpr and tpr for all thresholds of the classification
        if num_classes > 1:
            fpr, tpr, threshold = metrics.roc_curve(test_label[:, cl], preds_prob[:, cl])
        else:
            fpr, tpr, threshold = metrics.roc_curve(test_label, preds_prob)
        roc_auc = metrics.auc(fpr, tpr)
        # plot auc
        plt.plot(fpr, tpr,
                 label='{}AUC={}'.format(target_names[cl] + ' ' if num_classes > 1 else '', round(roc_auc, 2)))

    plt.plot([0, 1], [0, 1], 'r--')

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.title('Receiver Operating Characteristic')

    plt.legend(loc='lower right')

    plt.show()
