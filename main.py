import numpy as np
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
import ghostml
from classification import *
from run_ghost import *
from run_smote import *
from run_svmrfe import *
from run_oxgboost import *
from run_ex_dep_cost import *
random_seed = 16

#Choose Data 


DEBUG = False
OUTLIER_REMOVAL = False
CROSS_VAL = True

def main(train_df, test_df, algorithm):
    
    #Split in labels and features for clarity
    labels_test = test_df['Class']
    features_test = test_df.drop(columns = ['Class'], axis = 1)
    labels_train = train_df['Class']
    features_train = train_df.drop(columns = ['Class'], axis = 1)

    max_ee = calculate_max_ee(test_df)
    
    #GHOST
    if algorithm == '1':
        #GHOST drops time column
        train_df = train_df.drop(columns=['Time'], axis = 1)
        features_train = features_train.drop(columns=['Time'], axis = 1)
        features_test = features_test.drop(columns=['Time'], axis = 1)
        test_df = test_df.drop(columns=['Time'], axis = 1)
        
        kappa,confusion,auc_tmp, EE, auprc = run_ghost(train_df, max_ee, features_train, labels_train, features_test, labels_test, "oxgb", random_seed)
        

    #SVMRFE
    if algorithm == '2':
        #Select the most predictive features (28 features as mentioned in the paper)
        kappa, confusion, auc_tmp, EE, auprc = run_svmrfe(features_train, max_ee, labels_train, features_test, labels_test)
        #features_train, labels_train = rfe.transform(features_train)#, labels_train)

    #SMOTE + CATBOOST
    if algorithm == '3':
        kappa, confusion, auc_tmp, EE, auprc = run_exp_dep_cost(features_train, max_ee, labels_train, features_test, labels_test)
        
    if algorithm == '4':
        kappa, confusion, auc_tmp, EE, auprc = run_oxgboost(features_train, max_ee, labels_train, features_test, labels_test)
        

        #Save results
        if not DEBUG:
            with open("results/results.txt", "w") as text_file:
                text_file.write("kappa: {}".format(kappa))
                text_file.write("\n---confusion matrix: {}".format(confusion))



    if DEBUG:
        return

    if CROSS_VAL:
        return  auc_tmp, EE, auprc


if __name__ == "__main__":
    #Read data

    DATA = input('Which dataset? (1) for Europe, (2) for PaySim\n')
    if DATA == '1':
        data_file_name = "data/creditcard.csv"
        df = read_csv_as_pd(data_file_name)

    if DATA == '2':
        data_file_name = "data/paysim_data_600.csv"
        df = read_csv_as_pd(data_file_name)
        df = preprocess_paysim(df)
        '''df = df.sample(n = 600000)
        df.rename(columns = {'amount':'Amount', "isFraud": "Class", "step": "Time"}, inplace = True)
        #df = one_hot_encoding(df)
        df_type = pd.get_dummies(df['type'])
        df_new = pd.concat([df, df_type], axis=1)

        df_new.to_csv('data/paysim_data_600.csv')
        '''



    #Remove outliers

    algorithm =  input('Which algorithm should be run? (1) GHOST, (2) svm-RFE, (3) CAT, (4) for OXGBoost\n')

    if DEBUG:
        df = df.sample(n = 10000)
    if OUTLIER_REMOVAL:
        df = delete_outliers(df)

    #Shuffle the dataframe
    #df = df.sample(frac=1).reset_index(drop = True)

    if CROSS_VAL == True and DEBUG == False:
        #apply cross validation
        folds = np.array_split(df, 5)
        indices = [0,1,2,3,4]
        auc_folds = []
        auprc_folds = []
        ee_folds = []
        results = [0,0,0] #[AUROC, EE, AUPRC]
        for test_index in indices:
            print('------fold ', test_index, '---------')
            train_indices = np.delete(indices, test_index)
            train_df = pd.concat([folds[0], folds[1], folds[2], folds[3]])
            test_df = folds[test_index]

            #Run main
            auc, EE, auprc = main(train_df, test_df, algorithm)
            auc_folds = np.append(auc_folds, [auc])
            auprc_folds = np.append(auprc_folds, [auprc])
            ee_folds = np.append(ee_folds, [EE])
            results[0] = results[0] + (auc / 5.0)
            results[1] = results[1] + (EE / 5.0)
            results[2] = results[2] + (auprc / 5.0)

        std_auc = np.std(auc_folds)
        std_ee = np.std(ee_folds)
        std_auprc = np.std(auprc_folds)

        print_summary(algorithm, DATA)

        print('Cross validation report: ')
        print('Average auc = ', "{:.4f}".format(results[0]), ' +- ', "{:.4f}".format(std_auc))
        print('Average EE = ', "{:.4f}".format(results[1]), ' +- ', "{:.4f}".format(std_ee))
        print('Average auprc = ', "{:.4f}".format(results[2]), ' +- ', "{:.4f}".format(std_auprc))


    if CROSS_VAL == False or DEBUG == True:
        train_df, test_df = train_test_split(df, test_size=0.2)
        main(train_df, test_df, algorithm)
