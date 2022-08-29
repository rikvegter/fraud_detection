import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from classification import *
from run_smote import *
from utils import *

def preprocessing(df):
    '''
    This function scales the amount feature of a dataframe
    
    :param df: Pandas dataframe containing features
    :returns: Pandas dataframe with the scaled amount feature
    '''
    
    #Scale the time and amount
    scaler = StandardScaler()
    #df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    df[['Amount']] = scaler.fit_transform(df[['Amount']])
    return df

def run_exp_dep_cost(features_train, max_ee, labels_train, features_test, labels_test):
    '''
    This function runs the CatBoost + SMOTE algorithm and returns the evaluation metrics
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

    #Save the amount of money of the transactions before preprocessing
    amounts_of_transactions_train = features_train['Amount'].to_numpy()
    amounts_of_transactions_test = features_test['Amount'].to_numpy()
    #sum = np.sum(amounts_of_transactions_test)
    sum = max_ee
    test_amount = amounts_of_transactions_test

    #Apply preprocessing
    features_train = preprocessing(features_train)
    features_test = preprocessing(features_test)

    #Apply SMOTE
    features_train, labels_train = run_smote(features_train, labels_train)
    #TODO: FIX SAMPLING STRATEGY
    oversample = SMOTE(sampling_strategy = 'auto', random_state = None)
    features_train, labels_train = oversample.fit_resample(features_train, labels_train)

    #Train Catboost classifier
    clf = train_CB(features_train, labels_train)

    #Predict test set
    pred_test = clf.predict(features_test)
    probs_test = clf.predict_proba(features_test)[:,1]



    kappa, confusion, auc_tmp, EE, auprc = calc_metrics(features_test, labels_test, probs_test, pred_test, sum, test_amount)


    #----------------------------------------------#
    return kappa, confusion, auc_tmp, EE, auprc
