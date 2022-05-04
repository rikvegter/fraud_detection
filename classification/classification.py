import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

#Train a RF classifier and return the classifier
def train_RF(data, NUM_OF_TREES):
    #Split data and labels
    labels = data['Class']
    features = data.drop(columns = ['Class'], axis = 1)
    labels = labels.to_numpy()

    clf = RandomForestClassifier(n_estimators = NUM_OF_TREES, oob_score = True,
                                min_samples_leaf = 2, max_depth = 15)
    clf.fit(features, labels)
    return clf
