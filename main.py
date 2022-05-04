import numpy as np
import pandas as pd
from utilities.utils import *
from sklearn.model_selection import train_test_split
import ghostml
from classification.classification import train_RF
from run_ghost import run_ghost
from run_smote import run_smote

random_seed = 16

data_file_name = "data/creditcard.csv"
DEBUG = True
USE_GHOST = True
USE_PCA = False
PCA_components = 28
def main():
    #Read data
    df = read_csv_as_pd(data_file_name)

    if DEBUG:
        df = df.sample(n = 10000)


    #Split data in 80%-20% train-test data
    train_df, test_df = train_test_split(df, test_size=0.2)
    #Split in labels and features for clarity
    labels_test = test_df['Class']

    features_test = test_df.drop(columns = ['Class'], axis = 1)
    labels_train = train_df['Class']
    features_train = train_df.drop(columns = ['Class'], axis = 1)

    if USE_PCA:
        print("Running PCA with", PCA_components, 'components')
        features_train = apply_pca(features_train, PCA_components)
        features_test = apply_pca(features_test, PCA_components)


    if USE_GHOST:
        #Run the GHOST protocol
        kappa,confusion,auc_tmp = run_ghost(train_df, features_train, labels_train, features_test, labels_test, "RF", random_seed)


        #Save results
        if not DEBUG:
            with open("results/results.txt", "w") as text_file:
                text_file.write("kappa: {}".format(kappa))
                text_file.write("\n---confusion matrix: {}".format(confusion))


if __name__ == "__main__":
    main()
