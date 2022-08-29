from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def run_smote(features, labels):
    '''
    This functions applies SMOTE to a dataset and balances it to have equal sample size
    :param features: Pandas dataframe containing the features
    :param labels: Pandas dataframe containing the labels
    :return balanced_features: Pandas dataframe containing the features after SMOTE
    :return balanced_labels: Pandas dataframe containing the labels after SMOTE
    '''
    #feature_columns = features.columns
    og_features = features
    og_labels = labels
    #Create SMOTE object
    oversample = SMOTE(sampling_strategy = 'auto', random_state = None,
                        k_neighbors=5,n_jobs = 1)
    #Apply SMOTE
    balanced_features, balanced_labels = oversample.fit_resample(features, labels)

    #balanced_features = pd.DataFrame(balanced_features, columns=feature_columns)


    return balanced_features, balanced_labels
