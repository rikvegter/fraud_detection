import ghostml
from classification.classification import train_RF
import numpy as np
import pandas as pd

def obtain_threshold_train_subset(clf, data,
                    ThOpt_metrics = 'Kappa', N_subsets = 100,
                    subsets_size = 0.2, with_replacement = False,
                    random_seed = None):

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
