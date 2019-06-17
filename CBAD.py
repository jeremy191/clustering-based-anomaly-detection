#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:24:12 2019

@author: jeremyperez
"""
#%reset -f
#Main Libraries for the Project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Reading the Dataset
trainData = pd.read_csv("/Users/jeremyperez/Jupyter/NSL-KDD/KDDTrain+.csv", header = None) 
testData = pd.read_csv("/Users/jeremyperez/Jupyter/NSL-KDD/KDDTest+.csv", header = None)

#Getting the Dependent and independent Variables
X = trainData.iloc[:,:-2].values # Get all the rows and all the clums except all the colums - 2
Y = trainData.iloc[:,41].values# Get all the rows and the colum number 42

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


#We use dummy Encoding to pervent the machine learning don't atribute the categorical data in order. 

# Encoding the categorical data
# Encoding the Independent Variable
labelencoder_X1 = LabelEncoder()
labelencoder_X2 = LabelEncoder()
labelencoder_X3 = LabelEncoder()

X[:,1] = labelencoder_X1.fit_transform(X[:,1])
X[:,2] = labelencoder_X2.fit_transform(X[:,2])
X[:,3] = labelencoder_X3.fit_transform(X[:,3])

#Dummy encoding
onehotencoder = OneHotEncoder(categories = 'auto')
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)



#Elbow method to find the best number of culster
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#5 clusters 

#Applying K-mea(n_clusters = 5)
kmeans = KMeans(n_clusters = 5, init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Visual representation of the clusters

plt.scatter(X[y_kmeans ==0,0],X[y_kmeans == 0,1], s = 21, c = 'red', label = 'cluster1')
plt.scatter(X[y_kmeans ==1,0],X[y_kmeans == 1,1], s = 21, c = 'yellow', label = 'cluster2')
plt.scatter(X[y_kmeans ==2,0],X[y_kmeans == 2,1], s = 21, c = 'cyan', label = 'cluster3')
plt.scatter(X[y_kmeans ==3,0],X[y_kmeans == 3,1], s = 21, c = 'orange', label = 'cluster4')
plt.scatter(X[y_kmeans ==4,0],X[y_kmeans == 4,1], s = 21, c = 'black', label = 'cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1],s = 300, c = 'purple', label = 'Centroids')
plt.title('Clusters of Attacks')
plt.xlabel('Numbers of Attacks')
plt.ylabel('Types of Attacks')
plt.legend()
plt.show()
