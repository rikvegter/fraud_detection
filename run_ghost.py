from ghost.ghost import obtain_threshold_train_subset
from utils import *
from classification import *
import numpy as np

NUM_OF_TREES = 250

def run_ghost(train_df, max_ee, features_train, labels_train, features_test, labels_test, classifier, random_seed):
        '''
    This function runs the GHOST algorithm and returns the evaluation metrics
    :param features_train: Pandas dataframe containing the features of the training data
    :param max_ee: Float containing the maximum economic as a result of perfect classification
    :param labels_train: Pandas Series with the labels of the training data indicating whether transactions are fraud or genuine
    :param classifier: String indicating which classifier should be used. Options are "oxgb" and "RF"
    :param features_test: Pandas dataframe containing the features of the test data
    :param labels_test: Pandas Series with the labels of the test data indicating whether transactions are fraud or genuine
    :returns kappa: Float representing cohen's kappa based on the test labels and the predicted test labels
    :returns confusion: Numpy matrix with confusion matrix based on the test labels and the predicted test labels
    :returns auc_tmp: Float with the AUROC obtained by the classifier
    :returns EE: Float containing the economic efficiency as proposed in the thesis
    :returns auprc: Float representing the AUPRC obtained by the classifier
    '''
    
    #sum = features_test['Amount'].sum()
    sum = max_ee
    amounts_of_transactions = features_test['Amount'].to_numpy()
    
    
    
    #Train RF on training set
    if classifier == "RF":
        clf = train_RF(train_df, NUM_OF_TREES)
    if classifier == "oxgb":
        clf = train_oxgb(features_train, labels_train)
    if classifier == "cb":
        clf = train_CB(features_train, labels_train)

    #Obtain the best threshold according to GHOST procedure
    thresh_sub = obtain_threshold_train_subset(clf, train_df, random_seed = random_seed)
    
    #thresh_sub = 0.5
    print('decision threshold = ', thresh_sub)

    #Attach probabilities to test set
    probs_test = clf.predict_proba(features_test)[:,1]
    
    features_test['Utility'] = probs_test * features_test['Amount']

    #scores = [1 if x>=thresh_sub else 0 for x in probs_test]
    # calculate metrics using the optimized decision threshold
    labels = provide_classification_labels(thresh_sub, features_test, probs_test)
    kappa, confusion, auc_tmp, EE, auprc = calc_metrics(features_test, labels_test, probs_test, labels, sum, amounts_of_transactions, threshold = thresh_sub)

    #archive[assay_id].append(('GHOST',thresh_sub,kappa,confusion,auc_tmp))


    return kappa,confusion,auc_tmp, EE, auprc
