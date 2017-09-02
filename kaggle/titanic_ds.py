#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 19:14:35 2017

@author: yoovrajshinde
"""

# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from statistics import mode, mean

## importing the dataset
dataset = pd.read_csv('train.csv')
print("Checking NaN columns")
print(dataset.isnull().any())
dataset['Embarked'] = dataset['Embarked'].fillna(mode(dataset['Embarked']))
dataset['Age'] = dataset['Age'].fillna(mean(dataset['Age'][dataset['Age'].notnull()]))

print("Checking NaN columns After the cleaning")
print(dataset.isnull().any())

## X is matrix of features
X_train = dataset.iloc[:, [2,4,5,6,7,9, 11]].values

## Y is vector
Y_train = dataset.iloc[:, 1].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X_train[:, 1] = labelencoder_X1.fit_transform(X_train[:, 1])

labelencoder_X6 = LabelEncoder()
X_train[:, 6] = labelencoder_X6.fit_transform(X_train[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
X_train = onehotencoder.fit_transform(X_train).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)




# Fitting Classifier model to the dataset
# Create a classifier here
#from sklearn.linear_model import LogisticRegression
#regressor = LogisticRegression(solver='newton-cg', random_state=0)

from sklearn.neighbors import KNeighborsClassifier
regressor = KNeighborsClassifier(weights='uniform', 
                                 algorithm='auto',
                                 n_neighbors=50, 
                                 metric='minkowski', 
                                 p=2)

"""
# Fitting Classifier model to the dataset
# Create a classifier here
from sklearn.svm import SVC
regressor = SVC(kernel='rbf', degree=2,probability=True)
regressor.fit(X_train, Y_train)
"""

#from sklearn.tree import DecisionTreeClassifier
#regressor = DecisionTreeClassifier(random_state=0)

# Fitting Classifier model to the dataset
# Create a classifier here
#from sklearn.naive_bayes import GaussianNB
#regressor = GaussianNB()
#regressor.fit(X_train, Y_train)


regressor.fit(X_train, Y_train)



""" -------------  TEST DATA ----------- """
## importing the dataset
dataset = pd.read_csv('test.csv')
print("Checking NaN columns")
print(dataset.isnull().any())
dataset['Embarked'] = dataset['Embarked'].fillna(mode(dataset['Embarked']))
dataset['Age'] = dataset['Age'].fillna(mean(dataset['Age'][dataset['Age'].notnull()]))
dataset['Fare'] = dataset['Fare'].fillna(mean(dataset['Fare'][dataset['Fare'].notnull()]))

print("Checking NaN columns After the cleaning")
print(dataset.isnull().any())

## X is matrix of features
X_test = dataset.iloc[:, [1,3,4,5,6,8, 10]].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X_test[:, 1] = labelencoder_X1.transform(X_test[:, 1])

X_test[:, 6] = labelencoder_X6.transform(X_test[:, 6])
X_test = onehotencoder.transform(X_test).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
X_test = sc_X.transform(X_test)








## predicting a new result with logistic regression
Y_pred = regressor.predict(X_test)

dataset = pd.read_csv('gender_submission.csv')
Y_test = dataset.iloc[:, 1].values
# Visualizing the polynomial regression
#plt.plot(Y_test, color = 'red')
#plt.plot(Y_pred, color = 'blue')
plt.plot(Y_pred-Y_test, color = 'black')
plt.show()


df_output = pd.DataFrame()
df_output['PassengerId'] = dataset.iloc[:,0].values
df_output['Survived'] = Y_pred
df_output[['PassengerId','Survived']].to_csv('titanic_submit.csv',index=False)

# make a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

wrong_predictions = cm[0,1] + cm[1,0]
all_number = sum(sum(cm))
print(100 - 100*wrong_predictions / all_number)



"""
# Visualizing the Training set result
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:,0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:,1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set == j,1],
                c = ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (training set')
plt.xlabel('age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# Visualizing the Test set result
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:,0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:,1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set == j,1],
                c = ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (test set')
plt.xlabel('age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""