from utils import *
import pandas as pd
import pdb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from classification import train_RF
import time
from sklearn.decomposition import PCA
import seaborn as sns

data_file_name_europe = "data/creditcard.csv"
data_file_name_paysim = "data/paysim_data_600.csv"
NUM_OF_TREES = 250
DEBUG = False

def hist_of_utils(data):
    """
    This function takes the entire dataframe and attaches a new pandas column to it containing the utility
    which is defined as the amount of the transaction * the probability of a transaction being fraud.
    The function also print a histogram of the utility of all transactions.
    param data: Pandas frame containing all the features and the label of the data
    return void:
    """
    print('Training RF')
    clf = train_RF(data, NUM_OF_TREES)

    print('Attaching proba to dataset')
    probs_test = clf.predict_proba(data.drop(columns = ['Class'], axis = 1))[:,1]
    data['Utility'] = probs_test * data['Amount']


    bins = [0.0, 0.01, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    #hist = np.histogram(data['Utility'], bins = bins)
    print(data['Utility'].value_counts(bins = bins))
    sum = data['Utility'].value_counts(bins = bins).sum()

    print('Capturing ', sum, ' / ', len(data))
    return

def hist_of_amounts(data):
    '''
    This function takes the features of the data and creates a histogram of it
    :param data: Pandas frame containing all the features of the data
    '''
    amount_s = data['Amount']
    amounts_of_trans = data['Amount'].to_numpy()
    #hist = np.histogram(amounts_of_trans, bins = 10)
    import pdb; pdb.set_trace()
    print('Length of dataframe', len(data))
    plt.hist(amounts_of_trans, bins = 100)

    plt.title("Histogram of amounts of transactions within the range of 0.00 - 100,000")
    plt.xlabel('Amount of transactions in Euros')
    plt.ylabel('Frequency')
    plt.savefig('results/amounts_of_transactions_paysim.png')
    return

def obtain_stats_of_features(features, labels):
    '''
    This function prints standard statistics about the data set such as the mean amount + SD, maximum and minimum amount and the start and end time.
    :param features: Pandas dataframe containing the features of the data
    :param labels: Pandas Series containing the labels of the data features
    '''
    std = features['Amount'].std()
    min = features['Amount'].min()
    mean = features['Amount'].mean()
    max = features['Amount'].max()

    min_time = features['Time'].min()
    max_time = features['Time'].max()



    print('standard deviation = ', std)
    print('lowest transaction = ', min)
    print('mean = ', mean)
    print('highest transaction = ', max)

    print('start time = ', min_time)
    print('end time = ', max_time)

    

def visualize_pca(features, labels):
    '''
    This function applies PCA and plots the data along the principle components
    :param features: Pandas dataframe containing the features of the data
    :param labels: Pandas Series containing the labels of the data
    '''
    pca = PCA()
    Xt = pca.fit_transform(features)
    plot = plt.scatter(Xt[:,0], Xt[:,1], c=labels)
    plt.legend(handles=plot.legend_elements()[0], labels = list(labels))
    plt.savefig('results/pca_plot_paysim.png')



    return
    
def plot_explained_variance(df):
    '''
    This function shows the explained variance per pca principle component
    :param df: Dataframe containing the features of the data
    '''
    features_train = StandardScaler().fit_transform(df)

    pca = PCA()
    principalComponents = pca.fit_transform(features_train)
    
    
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('results/pca_explained_variance_paysim.png')
    return
    
def delete_outliers(df):
    '''
    This function delete the outliers based on the amount of the transactions
    :param df: Pandas Dataframe containing the features of the data
    '''
    df = df[df['Amount'] < 10000]
    return df

def hist_fraud_time(df):
    '''
    This function plots the number of fraud transactions over time
    :param df: Pandas dataframe containing the features of the data. This df should also contain the class labels.
    '''
    #Filter data on frauds
    fraud_df = df[df['Class'] == 1]

    max_time = df['Time'].max()
    range_length = 10
    time_ticks = np.arange(0, max_time, 20)
    spacing = int(max_time / range_length)
    bins = np.linspace(0,max_time, num = spacing)
    cum_fraud = fraud_df['Time'].value_counts(bins = bins, sort = False)

    cum_fraud.plot(xticks=[])
    #ax = cum_fraud.plot.hist(bins=len(cum_fraud))
    plt.xlabel('Time')
    plt.xticks(time_ticks)
    plt.ylabel('Frequency of fraud transactions')
    plt.title('Number of fraud transactions over time')
    plt.savefig('results/fraud_over_time_paysim.png')
    


def main():


    df = read_csv_as_pd(data_file_name_paysim)
    df = preprocess_paysim(df)
    if DEBUG:
        df = df.sample( n = 100)
    features = df.drop(columns = ['Class'], axis = 1)
    labels = df['Class']
    
    #Either one of the lines below should be uncommented to visualize part of the exploratory analysis. 
    
    #visualize_pca(features, labels)
    #plot_explained_variance(features)
    #df_small_amount = df[df['Amount'] < 100000]
    #df_small_amount = df[df['Amount'] < 500]
    #hist_of_amounts(df_small_amount)
    #hist_fraud_time(df)
    #obtain_stats_of_features(features, labels)
    #new_df = delete_outliers(df)
    #hist_of_utils(df)


if __name__ == "__main__":
    st = time.time()
    main()
    run_time = time.time() - st
    print('execution time = ', run_time, 'seconds')
