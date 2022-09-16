import ghostml
from classification import train_RF
import numpy as np
import pandas as pd

def obtain_threshold_train_subset(clf, data,
                    ThOpt_metrics = 'Kappa', N_subsets = 100,
                    subsets_size = 0.2, with_replacement = False,
                    random_seed = None):
    '''
    This function obtains the best threshold based on according to the GHOST algorithm
    :param clf: Classifier object used for classification
    :param data: Pandas dataframe used for finding the threshold
    :param ThOpt_metrics: String which can be set to use a paramter optimization. Default value is 'Kappa'
    :param N_subsets: Integer representing the number of subsets that should be used to split the data in. Default is 100 which is optimal according to the original Ghost paper
    :param subsets_size: Float indicating the size of the subsets. Set to 0.2 by default which represents 20% of the total data as a subset
    :param with_replacement: Boolean indicating whether different subsets can have the same samples. This is dependent on subsets_size and N_subsets.
    :returns opt_thresh: Float which is the optimal threshold as found by the Ghost algorithm
    '''

    thresholds = np.arange(0.05, 0.95, 0.05)

    #Split in features and labels
    fps_train = data.drop(columns = ['Class'], axis = 1)
    labels_train = data['Class']

    #Attach probabilities to datapoints
    probs_train = clf.predict_proba(fps_train)[:,1]

    #Convert to list for formatting
    labels_train = labels_train.tolist()
    probs_train = probs_train.tolist()



    opt_thresh = ghostml.optimize_threshold_from_predictions(
                    labels_train, probs_train, thresholds,
                    ThOpt_metrics = ThOpt_metrics, N_subsets = N_subsets,
                    subsets_size = subsets_size, with_replacement=with_replacement,
                    random_seed = random_seed
                )
    return opt_thresh
