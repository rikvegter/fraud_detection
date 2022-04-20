import numpy as np
import pandas as pd
from preprocessing.read_data import read_csv_as_pd
from sklearn.model_selection import train_test_split
import ghostml
from ghost import obtain_threshold_train_subset

random_seed = 16
data_file_name = "data/creditcard.csv"

def main():
    #Read data
    df = read_csv_as_pd(data_file_name)
    #Split data in 80%-20% train-test data
    train_df, test_df = train_test_split(df, test_size=0.2)

    #Obtain the best threshold according to GHOST procedure
    thresh_sub = obtain_threshold_train_subset(train_df, random_seed = random_seed)
    
    #store predictions in dataframe
    scores = [1 if x>=thresh_sub else 0 for x in test_probs]
    df_preds['GHOST'] = scores
    # calculate metrics using the optimized decision threshold
    kappa, confusion, auc_tmp = calc_metrics(labels_test, test_probs, threshold = thresh_sub)
    archive[assay_id].append(('GHOST',thresh_sub,kappa,confusion,auc_tmp))




if __name__ == "__main__":
    main()
