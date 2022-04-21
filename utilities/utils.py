import pandas as pd
from sklearn import metrics

def read_csv_as_pd(filename):
    """
    Read a csv file and convert to a pandas dataframe

    :param filename: String of path and filename
    :returns: pandas dataframe containing the data in the csv file
    """
    data = pd.read_csv(filename)
    return data

def calc_metrics(labels_test, test_probs, threshold = 0.5):
    scores = [1 if x>=threshold else 0 for x in test_probs]
    auc = metrics.roc_auc_score(labels_test, test_probs)
    kappa = metrics.cohen_kappa_score(labels_test,scores)
    confusion = metrics.confusion_matrix(labels_test,scores, labels=list(set(labels_test)))
    print('thresh: %.2f, kappa: %.3f, AUC test-set: %.3f'%(threshold, kappa, auc))
    print(confusion)
    print(metrics.classification_report(labels_test,scores))
    return kappa, confusion, auc
