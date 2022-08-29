import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import catboost as ctb
from xgboost import XGBClassifier


def provide_classification_labels(threshold, features_test, test_probs):
    '''
    This function provides classification labels based on a value of the decision threshold

    :param threshold: Float containing the decision threshold obtain by the ghost procedure
    :param features_test: Pandas frame containing the features of the test set
    :param test_probs: Numpy array containing the probabilities of the test transactions being fraud

    :returns labels: numpy array containing the predicted labels of the test set
    '''
    #Reset index of labels so they are in accordance with the test probs
    #features_test = features_test.reset_index(drop = True)

    labels = [1 if x>=threshold else 0 for x in test_probs]

    return labels

def train_CB(features, labels):
    '''
    This function trains a CatBoost classification algorithm with default parameters and returns the trained clf.
    
    :param features: Pandas Frame containing the features of the data
    :param labels: Pandas Series containing the labels of the data
    
    :returns clf: Trained CatBoost classification object
    '''
    clf = ctb.CatBoostClassifier(silent = True)
    clf.fit(features, labels)

    return clf

def train_oxgb(features, labels):
    '''
    This function trains a XGBoost classification algorithm with optimized parameters and returns the trained clf.
    
    :param features: Pandas Frame containing the features of the data
    :param labels: Pandas Series containing the labels of the data
    
    :returns clf: Trained XGBoost classification object
    '''

    imbalance_ratio = labels.value_counts()[0] / labels.value_counts()[1]
    
    clf = XGBClassifier(n_estimators = 400, min_child_weight = 3, max_depth = 15, learning_rate = 0.3, gamma = 0.1, colsample_bytree = 0.4, scale_pos_weight = imbalance_ratio)
    
    clf.fit(features, labels)
    
    return clf


#Train a RF classifier and return the classifier
def train_RF(data, NUM_OF_TREES):
    '''
    This function trains a Random Forest classification algorithm with default parameters and returns the trained clf.
    
    :param features: Pandas Frame containing the features of the data
    :param labels: Pandas Series containing the labels of the data
    
    :returns clf: Trained Random Forest classification object
    '''

    #Split data and labels
    labels = data['Class']
    features = data.drop(columns = ['Class'], axis = 1)
    labels = labels.to_numpy()

    clf = RandomForestClassifier(n_estimators = NUM_OF_TREES, oob_score = True,
                                min_samples_leaf = 2, max_depth = 15)
    clf.fit(features, labels)
    return clf
