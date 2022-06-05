from src.evaluate.plot_cm import plot_confusion_matrix
from sklearn import metrics


def calc_metrics(test_label, preds):
    classes_names = ['0', '1', '2', '3', '4']

    plot_confusion_matrix(cm=metrics.confusion_matrix(test_label, preds),
                          target_names=classes_names,
                          normalize=False)

    print("Accuracy:",
          round(metrics.accuracy_score(test_label, preds), 5),
          '\nBalanced accuracy:',
          round(metrics.balanced_accuracy_score(test_label, preds), 5),
          '\nMulticlass f1-score:',
          '\n    micro:', round(metrics.f1_score(test_label, preds, average='micro'), 5),
          '\n    macro:', round(metrics.f1_score(test_label, preds, average='macro'), 5),
          '\n    weighted:', round(metrics.f1_score(test_label, preds, average='weighted'), 5))

    print('\n\nClassification report:\n')
    print(metrics.classification_report(test_label, preds, digits=5))
