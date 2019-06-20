#reset -f
#Main Libraries for the Project

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


#trainData = pd.read_csv("/Users/bethanydanner/Google_Drive/documents/python_code/clustering-based-anomaly-detection/Dataset/NSL-KDD/KDDTrain+.csv", header = None)
#testData = pd.read_csv("/Users/bethanydanner/Google_Drive/documents/python_code/clustering-based-anomaly-detection/Dataset/NSL-KDD/KDDTest+.csv", header = None) 

#Reading the Train Dataset and Checking if has missing Values
trainData = pd.read_csv("/Users/jeremyperez/Jupyter/NSL-KDD/KDDTrain+.csv", header = None) 
#Run a Missing Value Ratio test to determine if any feature is missing values.
#If all ratios = 0.0, then data is not missing any values for any features.
#More info about Missing value ratio at 
#https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/
trainData.isnull().sum()/len(trainData)*100


#Reading the Test Dataset and Checking if has missing Values
testData = pd.read_csv("/Users/jeremyperez/Jupyter/NSL-KDD/KDDTest+.csv", header = None)
#Run a Missing Value Ratio test to determine if any feature is missing values.
#If all ratios = 0.0, then data is not missing any values for any features.
testData.isnull().sum()/len(testData)*100


#Getting the Dependent and independent Variables
X = trainData.iloc[:,:-1].values # Get all the rows and all the clums except all the colums - 1
Y = trainData.iloc[:,42].values# Get all the rows and the colum number 42
A = testData.iloc[:,:-1].values # Get all the rows and all the clums except all the colums - 1
Z = testData.iloc[:,42].values# Get all the rows and the colum number 42
attacks = trainData.iloc[:,42].values #Attacks with no one hot encoding

Y = pd.DataFrame(Y)
Z = pd.DataFrame(Z)

#Encoding Categorical Data for Train Set
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#We use One hot encoding to pervent the machine learning to atribute the categorical data in order. 
#What one hot encoding(ColumnTransformer) does is, it takes a column which has categorical data, 
#which has been label encoded, and then splits the column into multiple columns.
#The numbers are replaced by 1s and 0s, depending on which column has what value
#We don't need to do a label encoded step because ColumnTransformer do one hot encode and label encode!

#Encoding the Independient Variable
transformX = ColumnTransformer([("Servers", OneHotEncoder(categories = "auto"), [1,2,3])], remainder="passthrough")
X = transformX.fit_transform(X)
#Encoding the Dependent Variable
transformY= ColumnTransformer([("Attacks", OneHotEncoder(categories = "auto"), [0])], remainder="passthrough")
Y = transformY.fit_transform(Y)



#Encoding Categorical Data for Test Set
#Encoding the Independient Variable
transformA = ColumnTransformer([("Servers", OneHotEncoder(categories = "auto"), [1,2,3])], remainder="passthrough")
A = transformA.fit_transform(A)
    
#Encoding the Dependent Variable
transformZ = ColumnTransformer([("Attacks", OneHotEncoder(categories = "auto"), [0])], remainder="passthrough")
Z = transformZ.fit_transform(Z)



#Because we are using numerical-value-only clustering techniques to analyze the NSL-KDD dataset,
#we need to normalize the values in the dataset, as Ibrahim., et. al. describe (page 112).
#We complete the normalization process below:
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression

normalizer = Normalizer().fit(X)
trainX = normalizer.transform(X)

normalizer = Normalizer().fit(A)
testA = normalizer.transform(A)


trainData = np.array(trainX)
trainLabel =np.array(Y)

testData =  np.array(testA)
testLabel = np.array(Z)

#model = LogisticRegression(solver = 'lbfgs')
#model.fit(trainData,trainLabel)

#Elbow Method
#trainData
#Elbow method to find the best number of culster
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
    kmeans.fit(trainData)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#5 clusters 

#KMeans
#Applying K-mea(n_clusters = 5)
KMEANS = KMeans(n_clusters = 2, init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
kmeans = KMEANS.fit(trainData)
kmeans.labels_
pd.crosstab(attacks,kmeans.labels_)
