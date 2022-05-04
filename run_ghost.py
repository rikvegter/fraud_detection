from ghost.ghost import obtain_threshold_train_subset
from utilities.utils import *
from classification.classification import train_RF
import numpy as np

NUM_OF_TREES = 500

def run_ghost(train_df, features_train, labels_train, features_test, labels_test, classifier, random_seed):
    """
    Run the GHOST procedure. This function returns the metrics of the classification

    :param train_df: Numpy 2D array consisting of the features and labels of the training dataset
    :param features_test: Numpy 2D array consisting of the features of the test dataset
    :param labels_test: Numpy 1D array consisting of the labels of the test dataset, (0 or 1)
    :param classifier: Specification of what classifier the GHOST procedure can take. Options are {"RF", ...}
    :param random_seed: Specify random seed to reproduce results
    :returns kappa: Cohen's kappa defines the performance of a classifier of an imbalanced dataset
    :returns confusion: Confusion matrix defining the number of TP, TN, FP, FN
    :returns auc_tmp: Area under the curve metrics
    """

    #Train RF on training set
    if classifier == "RF":
        clf = train_RF(train_df, NUM_OF_TREES)


    #Create a new train dataframe with the new dimensions
    #labels_np = labels_train.to_numpy()
    #train_df = features_train
    #labels_frame = pd.DataFrame(labels_np, columns = ['Class'])
    #train_df['Class'] = labels_frame

    #Obtain the best threshold according to GHOST procedure
    thresh_sub = obtain_threshold_train_subset(clf, train_df, random_seed = random_seed)

    #Attach probabilities to test set
    probs_test = clf.predict_proba(features_test)[:,1]
    import pdb; pdb.set_trace()
    #utility_scores = probs_test *

    #scores = [1 if x>=thresh_sub else 0 for x in probs_test]
    # calculate metrics using the optimized decision threshold

    kappa, confusion, auc_tmp = calc_metrics(features_test, labels_test, probs_test, threshold = thresh_sub)
    #archive[assay_id].append(('GHOST',thresh_sub,kappa,confusion,auc_tmp))


    return kappa,confusion,auc_tmp
