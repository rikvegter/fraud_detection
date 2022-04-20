import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#Train a RF classifier and return the classifier
def train_RF(data, labels):
    clf = RandomForestClassifier(n_estimators = 500, oob_score = True,
                                min_samples_leaf = 2)
    clf.fit(data, labels)
    return clf
