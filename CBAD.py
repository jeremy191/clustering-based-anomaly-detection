#@authors: jeremyperez,bethanydanner

#reset -f

import numpy as np
import pandas as pd 
#import matplotlib.pyplot as plt


def readingData(path):
    
    #Reading the Train Dataset
    
    #trainData = pd.read_csv("/Users/bethanydanner/Google_Drive/documents/python_code/clustering-based-anomaly-detection/Dataset/NSL-KDD/KDDTrain+.csv", header = None)
    dataSet = pd.read_csv(path, header = None)
    
    return dataSet
#########################################################################
#trainData = pd.read_csv("/Users/bethanydanner/Google_Drive/documents/python_code/clustering-based-anomaly-detection/Dataset/NSL-KDD/KDDTrain+.csv", header = None)
dataSet = readingData("/Users/jeremyperez/Jupyter/NSL-KDD/KDDTrain+.csv")

#Run a Missing Value Ratio test to determine if any feature is missing values.
#If all ratios = 0.0, then data is not missing any values for any features.
dataSet.isnull().sum()/len(dataSet)*100
#########################################################################




#Getting The data we want to test for the clustering algorithms
def gettingVariables(dataSet):
    #Getting the Dependent and independent Variables
    X = dataSet.iloc[:,:-2].values # Data, Get all the rows and all the clums except all the colums - 2
    Y = dataSet.iloc[:,42].values# Labels

    #Removing Categorical data from the data set
    Z = dataSet.iloc[:,[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]].values
    
    #Removing server types
    W = dataSet.iloc[:,[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]].values
    
    #Removing Protocols to start using risk Values
    R = dataSet.iloc[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]].values

    return X,Y,Z,W,R
#########################################################################
data,labels,noCatg,noServ,riskVal  = gettingVariables(dataSet) #Getting the Data we want to use for the algorithms
#########################################################################



def encodingLabels(labels):
    #Binary Categories
    attackType  = {'normal': "normal", 'neptune': "abnormal", 'warezclient': "abnormal", 'ipsweep': "abnormal",'back': "abnormal", 'smurf': "abnormal", 'rootkit': "abnormal",'satan': "abnormal", 'guess_passwd': "abnormal",'portsweep': "abnormal",'teardrop': "abnormal",'nmap': "abnormal",'pod': "abnormal",'ftp_write': "abnormal",'multihop': "abnormal",'buffer_overflow': "abnormal",'imap': "abnormal",'warezmaster': "abnormal",'phf': "abnormal",'land': "abnormal",'loadmodule': "abnormal",'spy': "abnormal",'perl': "abnormal"} 
    attackEncodingCluster  = {'normal': 0,'abnormal': 1}
    
    labels[:] = [attackEncodingCluster[item] for item in labels[:]]#Changing the names of the labels to binary labels normal and abnormal
    labels[:] = [attackType[item] for item in labels[:]] #Encoding the binary data

    #4 Main Categories
    #attackType  = {'normal': "normal", 'neptune': "DoS", 'warezclient': "R2L", 'ipsweep': "Probe",'back': "DoS", 'smurf': "DoS", 'rootkit': "U2R",'satan': "Probe", 'guess_passwd': "R2L",'portsweep': "Probe",'teardrop': "DoS",'nmap': "Probe",'pod': "DoS",'ftp_write': "R2L",'multihop': "R2L",'buffer_overflow': "U2R",'imap': "R2L",'warezmaster': "R2L",'phf': "R2L",'land': "DoS",'loadmodule': "U2R",'spy': "R2L",'perl': "U2R"} 
    #attackEncodingCluster  = {'normal': 0,'DoS': 1,'Probe': 2,'R2L': 3, 'U2R': 4} #Main Categories
    
    #labels[:] = [attackEncodingCluster[item] for item in labels[:]]# Changing the names of attacks into 4 main categories
    #labels[:] = [attackType[item] for item in labels[:]] #Encoding the main 4 categories
    
    #normal = 0
    #DoS = 1
    #Probe = 2
    #R2L = 3
    #U2R = 4
    
    return labels

#########################################################################
labels = encodingLabels(labels)
#########################################################################





#Encoding the data using one hot encoding and using Main attacks categories or binary categories
def oneHotEncodingData(data): 
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    #We use One hot encoding to pervent the machine learning to atribute the categorical data in order. 
    #What one hot encoding(ColumnTransformer) does is, it takes a column which has categorical data, 
    #which has been label encoded, and then splits the column into multiple columns.
    #The numbers are replaced by 1s and 0s, depending on which column has what value
    #We don't need to do a label encoded step because ColumnTransformer do one hot encode and label encode!

    #Encoding the Independient Variable
    transform = ColumnTransformer([("Servers", OneHotEncoder(categories = "auto"), [1,2,3])], remainder="passthrough")
    data = transform.fit_transform(data)
    return data
#########################################################################
data= oneHotEncodingData(data) #One hot Encode with the complete data
#noServ = encodingData(noServ) #One hot Encode with no Server Type
#########################################################################




def riskEncodingData(data,labels): #This function is only for risk testing only
    #Manually Encoding for the attacks types only
    data = pd.DataFrame(data)
    servers  = {'http': 0.01, 'domain_u': 0, 'sunrpc': 1, 'smtp': 0.01, 'ecr_i': 0.87, 'iso_tsap': 1, 'private': 0.97, 'finger': 0.27, 'ftp': 0.26, 'telnet': 0.48,'other': 0.12,'discard': 1, 'courier': 1, 'pop_3': 0.53, 'ldap': 1, 'eco_i': 0.8, 'ftp_data': 0.06, 'klogin': 1, 'auth': 0.31, 'mtp': 1, 'name': 1, 'netbios_ns': 1,'remote_job': 1,'supdup': 1,'uucp_path': 1,'Z39_50': 1,'csnet_ns': 1,'uucp': 1,'netbios_dgm': 1,'urp_i': 0,'domain': 0.96,'bgp':1,'gopher': 1,'vmnet': 1,'systat': 1,'http_443': 1,'efs': 1,'whois': 1,'imap4': 1,'echo': 1,'link': 1,'login': 1,'kshell': 1,'sql_net': 1,'time': 0.88,'hostnames': 1,'exec': 1,'ntp_u': 0,'nntp': 1,'ctf': 1,'ssh': 1,'daytime': 1,'shell': 1,'netstat': 1,'nnsp': 1,'IRC': 0,'pop_2': 1,'printer': 1,'tim_i': 0.33,'pm_dump': 1,'red_i': 0,'netbios_ssn': 1,'rje': 1,'X11': 0.04,'urh_i': 0,'http_8001': 1,'aol': 1,'http_2784': 1,'tftp_u': 0,'harvest': 1}
    data[1] = [servers[item] for item in data[1]]

    servers_error  = {'REJ': 0.519, 'SF': 0.016, 'S0': 0.998, 'RSTR': 0.882, 'RSTO': 0.886,'SH': 0.993,'S1': 0.008,'RSTOS0': 1,'S3': 0.08,'S2': 0.05,'OTH': 0.729} 
    data[2] = [servers_error[item] for item in data[2]]

    #Attacks
    attackType  = {'normal': "normal", 'neptune': "abnormal", 'warezclient': "abnormal", 'ipsweep': "abnormal",'back': "abnormal", 'smurf': "abnormal", 'rootkit': "abnormal",'satan': "abnormal", 'guess_passwd': "abnormal",'portsweep': "abnormal",'teardrop': "abnormal",'nmap': "abnormal",'pod': "abnormal",'ftp_write': "abnormal",'multihop': "abnormal",'buffer_overflow': "abnormal",'imap': "abnormal",'warezmaster': "abnormal",'phf': "abnormal",'land': "abnormal",'loadmodule': "abnormal",'spy': "abnormal",'perl': "abnormal"} 
    #attackType  = {'normal': "normal", 'neptune': "DoS", 'warezclient': "R2L", 'ipsweep': "Probe",'back': "DoS", 'smurf': "DoS", 'rootkit': "U2R",'satan': "Probe", 'guess_passwd': "R2L",'portsweep': "Probe",'teardrop': "DoS",'nmap': "Probe",'pod': "DoS",'ftp_write': "R2L",'multihop': "R2L",'buffer_overflow': "U2R",'imap': "R2L",'warezmaster': "R2L",'phf': "R2L",'land': "DoS",'loadmodule': "U2R",'spy': "R2L",'perl': "U2R"} 
    labels[:] = [attackType[item] for item in labels[:]]
    
    #attackEncodingCluster  = {'normal': 0,'DoS': 1,'Probe': 2,'R2L': 3, 'U2R': 4} #Main Categories
    attackEncodingCluster  = {'normal': 0,'abnormal': 1}  #Binary Categories
    labels[:] = [attackEncodingCluster[item] for item in labels[:]]
    
    return data,labels
#########################################################################
riskVal,labels = riskEncodingData(riskVal,labels)
#########################################################################




def normalizing(data): #Scalign the data with the normalize method
    
    from sklearn.preprocessing import Normalizer
    #Because we are using numerical-value-only clustering techniques to analyze the NSL-KDD dataset,
    #we need to normalize the values in the dataset, as Ibrahim., et. al. describe (page 112).
    #Normalize works by scaling the features in a range of [0,1]
    #We complete the normalization process below:
    normalizer = Normalizer().fit(data)
    data = normalizer.transform(data)
    data = pd.DataFrame(data)
    
    return data
#########################################################################
#data = normalizing(data) #CategoricalData
noCatg = normalizing(noCatg) #No categorical data
#noServ = normalizing(noServ) #No Server Type
#riskVal = normalizing(riskVal) #Risk values with no protocols colum
#########################################################################



def featureSelection(data):
    from sklearn.feature_selection import VarianceThreshold
    
    selection = noCatg[[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]
    selector = VarianceThreshold()#You can specify the treshold you want
    selection = selector.fit_transform(selection)
    return selection
#########################################################################
noCatg = featureSelection(noCatg) #Dimensionality reduction , low variance filter technique on no categorical data
#data = featureSelection(data) #Dimensionality reduction , low variance filter technique on no categorical data
#########################################################################



def kmeansClustering(data): #K-means algorithm 
    from sklearn.cluster import KMeans
    #KMeans algorithm
    KMEANS = KMeans(n_clusters = 5, init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
    kmeans = KMEANS.fit(data)
    klabels = kmeans.labels_
    return klabels
#########################################################################
#KMEANS
klabels = kmeansClustering(data) #Categorical data Kmeans Algorithm
#klabels = kmeansClustering(noCatg) #No Categorical Data, Kmeans Algorithm
#klabels = kmeansClustering(noServ) #No server Type Data, Kmeans Algorithm
#klabels = kmeansClustering(riskVal) #Risk values with no protocols colum Data, Kmeans Algorithm


#Kmeans Results
kmeansR = pd.crosstab(labels,klabels)
kmeansR.idxmax()
#########################################################################





def kF1(klabels,labels): #F1 Score for Kmeans
    from sklearn.metrics import f1_score
    #Encoding data to F-score
    #normal = 0
    #DoS = 1
    #Probe = 2
    #R2L = 3
    #U2R = 4
    attackEncodingCluster  = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
    klabels[:] = [attackEncodingCluster[item] for item in klabels[:]]
    
    labels = np.array(labels,dtype = int)
    f1 = f1_score(labels,klabels, average = 'weighted') #[None, 'micro', 'macro', 'weighted']
    print(f1)
    
    return f1
#########################################################################
#F1 Score kmeans
kmeansF1 = kF1(klabels,labels)
kmeansF1
#########################################################################




def dbscanClustering(data): #DBSCAN algorithm
    from sklearn.cluster import DBSCAN
    
    #Compute DBSCAN
    db = DBSCAN(eps=0.2, min_samples = 1500).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    dblabels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(dblabels)) - (1 if -1 in dblabels else 0)
    n_noise_ = list(dblabels).count(-1)
    return dblabels,n_clusters_,n_noise_
#########################################################################
#DBSCAN
#dblabels = dbscanClustering(data) #Categorical Data DBSCAN Algorithm
dblabels,nClusters,nNoises = dbscanClustering(noCatg) #No Categorical Data, DBSCAN Algorithm
#dblabels,nClusters,nNoises = dbscanClustering(noServ) #No Server Type Data, DBSCAN Algorithm
#dblabels,nClusters,nNoises = dbscanClustering(riskVal) #Risk values with no protocols colum Data,DBSCAN Algorithm


#DBSCAN Results
dbscanR = pd.crosstab(labels,dblabels)
dbscanR.idxmax()
#########################################################################



def dbF1(dblabels,labels): #F1 score for DBSCAN
    from sklearn.metrics import f1_score
    #Encoding data to F-score
    #normal = 0
    #DoS = 1
    #Probe = 2
    #R2L = 3
    #U2R = 4
    attackEncodingCluster  = {-1: 1, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,7:1}
    dblabels[:] = [attackEncodingCluster[item] for item in dblabels[:]]
    
    labels = np.array(labels,dtype = int)
    f1 = f1_score(labels,dblabels, average = 'weighted') #[None, 'micro', 'macro', 'weighted']
    
    return f1
#########################################################################
#F1 Score dbscan
dbscanF1 = dbF1(dblabels,labels)
dbscanF1
#########################################################################