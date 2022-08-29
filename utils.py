import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
from numpy import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
from classification import *
np.seterr(divide='ignore', invalid='ignore')
from sklearn.metrics import plot_roc_curve
from sklearn.preprocessing import LabelEncoder
#METRICS
PLOT_CURVES = True
    

def calculate_max_ee(test_df):
    '''
    This function calculates the maximum economic efficiency which is a result of perfect classification
    :param test_df: Pandas Dataframe containing the features and labels of the test data
    :returns max_ee: Float indicating the maximum econimic efficiency
    '''
    max_ee = 0.0
    for i in range(len(test_df)):
        if test_df.iloc[i]['Class'] == 0:
            max_ee += test_df.iloc[i]['Amount'] * 0.03
        if test_df.iloc[i]['Class'] == 1:
            max_ee -= 150.0
        
    return max_ee

def preprocess_paysim(df):
    '''
    This function drops a column for the paysim data set which is irrelevant due to high entropy
    :param df: Pandas dataframe containing the PaySim data set
    :returns df: Pandas Dataframe without the columns that are dropped.
    '''
    
    df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)
    return df


    
def create_components_names(num_of_components):
    '''
    This function create components names used for pca
    :param num_of_components: Integer indicating the number of components
    returns columns: Numpy array with strings which are the component names
    '''
    columns = []
    for i in range(0, num_of_components):
        columns.append('component_' + str(i))
    return columns

def delete_outliers(df):
    df = df[df['Amount'] < 10000]
    return df

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


def calculate_economic_efficiency(y_pred, y_true, amounts_of_transactions, total_amount):
    """
    This function can calculate the economic efficiency

    :param y_pred: Numpy array representing the predicted classes
    :param y_true: Numpy array representing the true classes
    :param amounts_of_transactions: Numpy array containing the amounts of the credit card transactions
    :returns: The total amount of money of fraud transactions
    that failed to be identified as fraud
    """

    a = 150.0
    b = 150.0
    gamma = 0.03
    delta = 0.97

    unq = np.array([x+2*y for x,y in zip(y_pred, y_true)])

    #Calculate the indices for all possible classifications (tp, tn, fp, fn)
    tp = np.array(np.where(unq == 3)).tolist()[0]
    fp = np.array(np.where(unq == 1)).tolist()[0]
    tn = np.array(np.where(unq == 0)).tolist()[0]
    fn = np.array(np.where(unq == 2)).tolist()[0]

    EE = 0.0 #Economic efficiency
    
    #Handle true positives
    for i in tp:
        EE += (-b)
    #Handle false negatives
    for i in fn:
        EE += (-amounts_of_transactions[i] * delta)
    #Handle false positives
    for i in fp:
        EE += (amounts_of_transactions[i] * gamma - a)
    #Handle true negatives
    for i in tn:
        EE += (amounts_of_transactions[i] * gamma)
    
    
    EE = EE / total_amount

    return EE


def print_summary(algorithm, data):
    '''
    This function lets the user choose a pipeline on a specific data set
    :param algorithm: String with the algorithm that should be used
    :param data: String which represents which data set should be used
    
    '''
    if data == '1':
        print('\nDataset: Europe')
    if data == '2':
        print('\nDataset: PaySim')
    if algorithm == '1':
        print('Algorithm: GHOST')
    if algorithm == '2':
        print('Algorithm: SVM-RFE')
    if algorithm == '3':
        print('Algorithm: SMOTE + Cat')
    return


def read_csv_as_pd(filename):
    """
    Read a csv file and convert to a pandas dataframe

    :param filename: String of path and filename
    :returns: pandas dataframe containing the data in the csv file
    """
    data = pd.read_csv(filename)
    return data

def calc_metrics(features_test, labels_test, test_probs, pred_test, sum, amounts_of_transactions, threshold = 0.5):
    '''
    This function calculates the relevant metrics based on predicted labels and true labels
    :param features_test: Pandas frame containing the features of the test data
    :param labels_test: Pandas series containing the labels of the test features
    :param test_probs: Numpy array containing the probabilitis of the predicted transaction. Every element represent the probability of a transaction being fraud according to the classifier.
    :param pred_test: Numpy array containing the predicted labels of the test features
    :param sum: Float which says the total amount of all transactions in the test set
    :param amounts_of_transactions: Numpy array containing the amount of the transactions in the test set
    :param threshold: Float which is the decision threshold. Default is set to 0.5.
    :returns kapa: Cohens kappa of the predicted labels and the true labels of the test set
    :returns confusion: Numpy matrix which represents the confusion matrix
    :returns auc: Float of the AUROC obtained from the predicted labels and the true labels on the test data
    :returns EE: Float of the Economic efficiency which represents the financial gain of a pipeline
    :returns auprc: Float representing the AUPRC of the predicted labels and the true labels on the test data
    '''

    #scores = provide_classification_labels(threshold, features_test, test_probs)
    scores = pred_test


    precision, recall, thresholds = precision_recall_curve(labels_test, test_probs)
    fscore = (2 * precision * recall) / (precision + recall)

    # locate the index of the largest f score
    ix = argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

    auprc = metrics.auc(recall, precision)

    auc = metrics.roc_auc_score(labels_test, test_probs)
    kappa = metrics.cohen_kappa_score(labels_test,scores)
    confusion = metrics.confusion_matrix(labels_test,scores, labels=list(set(labels_test)))
    fpr, tpr, thresholds = metrics.roc_curve(labels_test, test_probs, pos_label = 1)
    auc = metrics.auc(fpr, tpr)


    #Print the metrics
    print('---confusion matrix = ---\n', confusion)
    print('thresh: %.2f, kappa: %.3f, AUC test-set: %.10f, AUPRC test-set: %.10f'%(threshold, kappa, auc, auprc))
    print('Classification Report:\n')

    print(metrics.classification_report(labels_test,pred_test))

    #Obtain the total loss in dollars
    #amounts_of_transactions = features_test['Amount'].to_numpy()

    #sum = features_test['Amount'].sum())

    #amounts_of_transactions = features_test[:,-1]
    #sum = np.sum(amounts_of_transactions)
    EE = calculate_economic_efficiency(labels_test, scores, amounts_of_transactions, sum)
    print('\nThe EE =  ', "{:.4f}".format(EE))


    print('The total amount of money in the test set is', "{:.2f}".format(sum))

    #Plot ROC-curve
    if PLOT_CURVES:
        plot_roc_curve(fpr, tpr, auc)
        plot_pr_curve(precision, recall, auprc)

    return kappa, confusion, auc, EE, auprc

def plot_pr_curve(precision, recall, auprc):
    '''
    This function plots the precision recall curve
    :param precision: List showing the precision over various decision thresholds
    :param recall: List showing the recall over various decision thresholds
    :param auprc: float of the auprc.
    '''
    plt.title('Precision Recall Curve')
    plt.plot(recall, precision, 'b', label = 'AUPRC = %0.3f'%auprc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('results/eu_svmrfe_auprc.png')
    plt.figure().clear()


def plot_roc_curve(fpr, tpr, auc):
    '''
    This function plots the precision recall curve
    :param fpr: List showing the false positive rate over various decision thresholds
    :param tpr: List showing the true positive rate over various decision thresholds
    :param auc: float of the auc
    '''
    plt.title('Receiver Operating Characterstic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f'%auc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('results/eu_svmrfe_auroc.png')
    plt.figure().clear()











#----------------UNUSED FUNCTIONS------------------------#
def calculate_fraud_loss(y_pred, y_true, amounts_of_transactions, total_amount):
    """
    This function can calculate the loss of the fraud transactions that
    are not classified as fraud

    :param y_pred: Numpy array representing the predicted classes
    :param y_true: Numpy array representing the true classes
    :param amounts_of_transcations: Numpy array indicating the amount of the transactions.
    :param total_amount: Float showing the cumulative sum of all amounts of the transactions in the set
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


    alpha = 150
    total_loss += (len(tp) + len(fp)) * alpha

    fraud_fraction = (total_loss / total_amount ) * 100.0
    return fraud_fraction


def one_hot_encoding_columns(df):
    print('Start One-Hot-Encoding')
    df = pd.get_dummies(df)
    return df

def one_hot_encoding(df):
    '''
    This function uses one-hot-encoding applied on the
    '''
    le = LabelEncoder()
    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = le.fit_transform(df[column_name])
        else:
            pass
    return df
