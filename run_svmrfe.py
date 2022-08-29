import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from run_smote import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from utils import *
from sklearn.model_selection import StratifiedKFold
NUM_PRED_FEATURES = 6


def run_svmrfe(features_train, max_ee, labels_train, features_test, labels_test):
        '''
    This function runs the SVMRFE algorithm and returns the evaluation metrics
    :param features_train: Pandas dataframe containing the features of the training data
    :param max_ee: Float containing the maximum economic as a result of perfect classification
    :param labels_train: Pandas Series with the labels of the training data indicating whether transactions are fraud or genuine
    :param features_test: Pandas dataframe containing the features of the test data
    :param labels_test: Pandas Series with the labels of the test data indicating whether transactions are fraud or genuine
    :returns kappa: Float representing cohen's kappa based on the test labels and the predicted test labels
    :returns confusion: Numpy matrix with confusion matrix based on the test labels and the predicted test labels
    :returns auc_tmp: Float with the AUROC obtained by the classifier
    :returns EE: Float containing the economic efficiency as proposed in the thesis
    :returns auprc: Float representing the AUPRC obtained by the classifier
    '''
    
    #Apply SMOTE
    features_train, labels_train = run_smote(features_train, labels_train)
    
    #This function select the 28 most predictive features
    rfe = RFE(estimator = svm.LinearSVC(), n_features_to_select = NUM_PRED_FEATURES)
        #cv = StratifiedKFold(2))
    rfe.fit(features_train, labels_train)
    
    
    #Reduce dimensions

    sum = features_test['Amount'].sum()
    amounts_of_transactions = features_test['Amount'].to_numpy()

    features_train = rfe.transform(features_train)
    features_test = rfe.transform(features_test)
    print('selected features: ', len(features_train[0]))

    

    #Train RF
    clf = RandomForestClassifier(bootstrap = True, n_estimators = 100,
                                max_depth = 10, max_features = 2,
                                min_samples_leaf = 2, min_samples_split = 12,
                                max_leaf_nodes = 10)

    clf.fit(features_train, labels_train)

    #Predict the test set
    pred_test = clf.predict(features_test)
    probs_test = clf.predict_proba(features_test)[:,1]

    #Calculate metrics
    kappa, confusion, auc_tmp, EE, auprc = calc_metrics(features_test, labels_test, probs_test, pred_test, sum, amounts_of_transactions)

    return kappa,confusion,auc_tmp, EE, auprc
