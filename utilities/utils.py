import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def create_components_names(num_of_components):
    columns = []
    for i in range(0, num_of_components):
        columns.append('component_' + str(i))
    return columns

def apply_pca(features_train, PCA_components, plot = "off"):
    """
    This function applies PCA and outputs the explained variance, and plots
    this against the number of PCA components
    :param features_train: Pandas dataframe containing all predictors
    :param labels_train: Pandas dataframe containing all labels
    :returns principalDf: Dataframe containing the data with reduced dimensions
    """

    component_names = []
    for i in range(1, PCA_components):
        component_names.append("pca"+str(i))

    #Scale features
    features_train = StandardScaler().fit_transform(features_train)

    pca = PCA(PCA_components)
    principalComponents = pca.fit_transform(features_train)


    #Create dataframe with names
    component_names = create_components_names(PCA_components)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = component_names)


    if plot == "on":
        print(np.cumsum(pca.explained_variance_ratio_))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.savefig('results/pca_explained_variance.png')

    return principalDf


def calculate_fraud_loss(y_pred, y_true, amounts_of_transactions):
    """
    This function can calculate the loss of the fraud transactions that
    are not classified as fraud

    :param y_pred: Numpy array representing the predicted classes
    :param y_true: Numpy array representing the true classes
    :returns: The total amount of money of fraud transactions
    that failed to be identified as fraud
    """

    #Calculate the unique indices
    unq = np.array([x+2*y for x,y in zip(y_pred, y_true)])

    #Calculate the indices for all possible classifications (tp, tn, fp, fn)
    tp = np.array(np.where(unq == 3)).tolist()[0]
    fp = np.array(np.where(unq == 1)).tolist()[0]
    tn = np.array(np.where(unq == 0)).tolist()[0]
    fn = np.array(np.where(unq == 2)).tolist()[0]

    #Add all the losses to obtain a total loss
    total_loss = 0
    for index in fn:
        loss = amounts_of_transactions[index]
        total_loss += loss

    return total_loss

def read_csv_as_pd(filename):
    """
    Read a csv file and convert to a pandas dataframe

    :param filename: String of path and filename
    :returns: pandas dataframe containing the data in the csv file
    """
    data = pd.read_csv(filename)
    return data

def calc_metrics(features_test, labels_test, test_probs, threshold = 0.5):
    scores = [1 if x>=threshold else 0 for x in test_probs]
    auc = metrics.roc_auc_score(labels_test, test_probs)
    kappa = metrics.cohen_kappa_score(labels_test,scores)
    confusion = metrics.confusion_matrix(labels_test,scores, labels=list(set(labels_test)))
    fpr, tpr, thresholds = metrics.roc_curve(labels_test, test_probs, pos_label = 1)
    auc = metrics.auc(fpr, tpr)

    #Print the metrics
    print('---confusion matrix = ---\n', confusion)
    print('thresh: %.2f, kappa: %.3f, AUC test-set: %.3f'%(threshold, kappa, auc))
    print('Classification Report:\n')
    print(metrics.classification_report(labels_test,scores))

    #Obtain the total loss in dollars

    '''
    amounts_of_transactions = features_test['Amount'].to_numpy()
    fraud_loss = calculate_fraud_loss(labels_test, scores, amounts_of_transactions)
    print('\nThe total loss due to fraud transaction is ', "{:.2f}".format(fraud_loss))
    print('The total amount of money in the test set is', "{:.2f}".format(features_test['Amount'].sum()))


    fraud_fraction = (fraud_loss / features_test['Amount'].sum())*100
    print('Fraud loss fraction = ', "{:.2f}".format(fraud_fraction))
    '''

    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('results/auc_curve.png')

    return kappa, confusion, auc
