#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:02:04 2017

@author: yoovrajshinde
"""

# NLP

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
## ignoring the double quotes (quoting=3)
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
import re
# remove the non alphabets
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
# lower case
review = review.lower()
# convert to list
review = review.split()

# library for getting words information
import nltk
# download the stop words 
nltk.download('stopwords')

# import the stopwords from your nltk local folder  (downloaded from download command)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
# use set to allow python algorithms to process fast
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

review = ' '.join(review)

def clean(reviewString):
    reviewString = re.sub('[^a-zA-Z]', ' ', reviewString)
    reviewString = reviewString.lower()
    reviewString = reviewString.split()
    reviewString = [ps.stem(word) for word in reviewString if not word in set(stopwords.words('english'))]
    reviewString = ' '.join(reviewString)
    return reviewString

corpus=[]
for i in range(0,len(dataset)):
    corpus.append(clean(dataset['Review'][i]))
    

# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:,1].values

# ---------- Naive based ----------
def naiveBased():
    
    # Splitting data into training and testing data
    from sklearn.cross_validation import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
    
    
    # Fitting Classifier model to the dataset
    # Create a classifier here
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    
    ## predicting a new result with logistic regression
    Y_pred = classifier.predict(X_test)
    
    # Visualizing the polynomial regression
    plt.plot(Y_pred-Y_test, color = 'black')
    plt.show()
    
    
    
    # make a confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)
    #Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (cm[0][0] + cm[1][1])/sum(sum(cm))
    
    #Precision = TP / (TP + FP)
    precision = cm[0][0]/(cm[0][0] + cm[0][1])
    
    #Recall = TP / (TP + FN)
    recall = cm[0][0]/(cm[0][0] + cm[1][0])
    
    #F1 Score = 2 * Precision * Recall / (Precision + Recall)
    f1_score = 2 * precision * recall / (precision + recall)
    
    return (accuracy, precision, recall, f1_score)
    


# ---------- Decision trees ----------
def decisionTrees():
    # Splitting data into training and testing data
    from sklearn.cross_validation import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
    
    # Fitting Classifier model to the dataset
    # Create a classifier here
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, Y_train)
    
    ## predicting a new result with logistic regression
    Y_pred = classifier.predict(X_test)
    
    # Visualizing the polynomial regression
    plt.plot(Y_pred-Y_test, color = 'black')
    plt.show()
    
    # make a confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)
    #Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (cm[0][0] + cm[1][1])/sum(sum(cm))
    
    #Precision = TP / (TP + FP)
    precision = cm[0][0]/(cm[0][0] + cm[0][1])
    
    #Recall = TP / (TP + FN)
    recall = cm[0][0]/(cm[0][0] + cm[1][0])
    
    #F1 Score = 2 * Precision * Recall / (Precision + Recall)
    f1_score = 2 * precision * recall / (precision + recall)
    
    return (accuracy, precision, recall, f1_score)



# ------------ Random Forest -----------
def randomForest():
    # Splitting data into training and testing data
    from sklearn.cross_validation import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
    
    # Fitting Classifier model to the dataset
    # Create a classifier here
    from sklearn.ensemble import  RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, Y_train)
    
    
    ## predicting a new result with logistic regression
    Y_pred = classifier.predict(X_test)
    
    # Visualizing the polynomial regression
    plt.plot(Y_pred-Y_test, color = 'black')
    plt.show()
    
    # make a confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)
    #Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (cm[0][0] + cm[1][1])/sum(sum(cm))
    
    #Precision = TP / (TP + FP)
    precision = cm[0][0]/(cm[0][0] + cm[0][1])
    
    #Recall = TP / (TP + FN)
    recall = cm[0][0]/(cm[0][0] + cm[1][0])
    
    #F1 Score = 2 * Precision * Recall / (Precision + Recall)
    f1_score = 2 * precision * recall / (precision + recall)
    
    return (accuracy, precision, recall, f1_score)

print("accuracy, precision, recall, f1_score\n{}".format(naiveBased()))
naive_results = naiveBased()
decision_tree_results = decisionTrees()
random_forest_results = randomForest()
plt.plot(naive_results, color='red')
plt.plot(decision_tree_results, color='blue')
plt.plot(random_forest_results, color='green')
plt.legend()