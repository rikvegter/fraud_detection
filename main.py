import numpy as np
import pandas as pd
from utilities.utils import *
from sklearn.model_selection import train_test_split
import ghostml
from classification.classification import train_RF
from ghost import obtain_threshold_train_subset


random_seed = 16
data_file_name = "data/creditcard.csv"
DEBUG = False
def main():
    #Read data
    df = read_csv_as_pd(data_file_name)

    if DEBUG:
        df = df.sample(n = 5000)


    #Split data in 80%-20% train-test data
    train_df, test_df = train_test_split(df, test_size=0.2)
    #Split in labels and features for clarity
    labels_test = test_df['Class']
    features_test = test_df.drop(columns = ['Class'], axis = 1)
    labels_train = train_df['Class']
    features_train = train_df.drop(columns = ['Class'], axis = 1)

    #Train RF on training set
    clf = train_RF(train_df)

    #Obtain the best threshold according to GHOST procedure
    thresh_sub = obtain_threshold_train_subset(clf, train_df, random_seed = random_seed)

    #Attach probabilities to test set
    probs_test = clf.predict_proba(features_test)[:,1]


    scores = [1 if x>=thresh_sub else 0 for x in probs_test]
    # calculate metrics using the optimized decision threshold
    kappa, confusion, auc_tmp = calc_metrics(labels_test, probs_test, threshold = thresh_sub)
    #archive[assay_id].append(('GHOST',thresh_sub,kappa,confusion,auc_tmp))


    print('kappa = ', kappa)
    print('---confusion matrix = ---', confusion)





if __name__ == "__main__":
    main()
