import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from utils import *

def run_oxgboost(features_train, max_ee, labels_train, features_test, labels_test):
    '''
    This function runs the OXGBoost algorithm and returns the evaluation metrics
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
    
    #sum = features_test['Amount'].sum()
    sum = max_ee
    amounts_of_transactions = features_test['Amount'].to_numpy()

    #train XGBoost with optimized parameters (OXGBoost)
    imbalance_ratio = labels_train.value_counts()[0] / labels_train.value_counts()[1]
    
    
    clf = XGBClassifier(n_estimators = 400, min_child_weight = 3, max_depth = 15, learning_rate = 0.3, gamma = 0.1, colsample_bytree = 0.4, scale_pos_weight = imbalance_ratio)
    
    clf.fit(features_train, labels_train)
    
    pred_test = clf.predict(features_test)
    probs_test = clf.predict_proba(features_test)[:,1]
    
    #Calculate metrics
    kappa, confusion, auc_tmp, EE, auprc = calc_metrics(features_test, labels_test, probs_test, pred_test, sum, amounts_of_transactions)

    return kappa,confusion,auc_tmp, EE, auprc
    
