# Optimizing the financial gain for credit card fraud detection systems using machine learning techniques

This repository contains the pipelines used for repdocuding the results of my master thesis which is about optimizing financial gain for credit card fraud detection. 

**Setup:**
1. Clone the github repository
2. Download the zipped data from https://www.dropbox.com/home/Master%20Thesis%20data and unzip in same directory
2. Create a folder 'data' and put both data files in this folder.
3. Download requirements with the following command: pip3 install -r requirements.txt
4. Follow run instructions

**Run instructions:**
1. python3 main.py
2. Choose data set you want to test by typing 1 or 2 followed by 'enter'
3. Choose pipeline you want to test by typing 1, 2, 3 or 4 followed by 'enter'
---
## Abstract of Thesis
Credit card fraud detection is a worldwide problem that comes with significant financial losses for credit card service companies. Due to the high number of transactions that occur daily, there is a need for the automatization of the classification of these transactions using machine learning techniques. One of the main problems within credit card fraud detection is the imbalance between \textit{fraudulent} and \textit{genuine} transactions. Machine learning algorithms tend to be biased towards the majority class. Therefore data balancing techniques are required to overcome the problem of class imbalance. This thesis provides a comparison of state-of-the-art methods in terms of a new proposed, more realistic financial metric. The financial metric is defined based on expert advice from icscards and the literature of credit card fraud detection. While the financial metrics in literature only take into account a percentual gain and loss of transactions, this metric also takes external costs that credit card service companies make into account. Also, a class imbalance machine learning method used to optimize the decision threshold from another domain (called Ghost) where class imbalance occurs is trained on credit card fraud data. Ghost finds the optimal decision threshold combined with a Random Forest classifier. The results showed that boosting methods using implicit balancing techniques perform state-of-the-art results both in terms of the number of correct classifications and in terms of economic efficiency resulting in near-to-perfect classification. This study also analyzes the accepted loss in terms of AUROC/AUPRC of classifiers optimized on the economic efficiency and where the economic efficiency fits within the precision-recall trade-off. Due to the significant financial loss for false negatives, a slight decrease is found in economic efficiency for low precision. 

University of Groningen Master Thesis Artificial Intelligence
Author: Rik Vegter
Supervised by Prof. dr. L.R.B. Schomaker & Maruf Dhali
