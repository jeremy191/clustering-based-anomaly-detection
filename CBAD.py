reset -f
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

Y = pd.DataFrame(Y)
Y = np.array(Y)
Z = pd.DataFrame(Z)
Z = np.array(Z)

###############################################################################################
#removing Categorical data from the data set
X = trainData.iloc[:,[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]].values
attacks = trainData.iloc[:,42].values #Attacks with no one hot encoding
###############################################################################################


#Encoding Categorical Data for Train Set
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
#We use One hot encoding to pervent the machine learning to atribute the categorical data in order. 
#What one hot encoding(ColumnTransformer) does is, it takes a column which has categorical data, 
#which has been label encoded, and then splits the column into multiple columns.
#The numbers are replaced by 1s and 0s, depending on which column has what value
#We don't need to do a label encoded step because ColumnTransformer do one hot encode and label encode!

###############################################################################################################
#Manually encode
#col1 = {'tcp': 0,'udp': 1,'icmp': 2} 
#col2  = {'http': 0, 'domain_u': 1, 'sunrpc': 2, 'smtp': 3, 'ecr_i': 4, 'iso_tsap': 5, 'private': 6, 'finger': 7, 'ftp': 8, 'telnet': 9,'other': 10,'discard': 11, 'courier': 12, 'pop_3': 13, 'ldap': 14, 'eco_i': 15, 'ftp_data': 16, 'klogin': 17, 'auth': 18, 'mtp': 19, 'name': 20, 'netbios_ns': 21,'remote_job': 22,'supdup': 23,'uucp_path': 24,'Z39_50': 25,'csnet_ns': 26,'uucp': 27,'netbios_dgm': 28,'urp_i': 29,'domain': 30,'bgp':31,'gopher': 32,'vmnet': 33,'systat': 34,'http_443': 35,'efs': 36,'whois': 37,'imap4': 38,'echo': 39,'link': 40,'login': 41,'kshell': 42,'sql_net': 43,'time': 44,'hostnames': 45,'exec': 46,'ntp_u': 47,'nntp': 48,'ctf': 49,'ssh': 50,'daytime': 51,'shell': 52,'netstat': 53,'nnsp': 54,'IRC': 55,'pop_2': 56,'printer': 57,'tim_i': 58,'pm_dump': 59,'red_i': 60,'netbios_ssn': 61,'rje': 62,'X11': 63,'urh_i': 64,'http_8001': 65,'aol': 66,'http_2784': 67,'tftp_u': 68,'harvest': 69} 
#col3  = {'REJ': 0, 'SF': 1, 'S0': 2, 'RSTR': 3, 'RSTO': 4,'SH': 5,'S1': 6,'RSTOS0': 7,'S3': 8,'S2': 9,'OTH': 10} 
#col42  = {'normal': 0, 'neptune': 1, 'warezclient': 2, 'ipsweep': 3, 'mscan': 4, 'back': 5, 'smurf': 6, 'mailbomb': 7, 'apache2': 8, 'rootkit': 9,'back': 10,'satan': 11, 'processtable': 12, 'guess_passwd': 13, 'saint': 14,'portsweep': 15,'teardrop': 16,'nmap': 17,'pod': 18,'ftp_write': 19,'multihop': 20,'buffer_overflow': 21,'imap': 22,'warezmaster': 21,'phf': 22,'land': 23,'loadmodule': 24,'spy': 25,'perl': 26,'snmpgetattack': 27,'httptunnel': 28,'ps': 29,'snmpguess': 30,'named': 31,'sendmail':32,'xterm':33,'worm': 34,'xlock': 35,'xsnoop': 36,'sqlattack': 37,'udpstorm':38} 
#trainData.col_1 = [col1[item] for item in trainData.col_1]
#trainData.col_2 = [col2[item] for item in trainData.col_2]
#trainData.col_3 = [col3[item] for item in trainData.col_3]
#Y[0] = [col42[item] for item in Y[0]]
###############################################################################################################


#Encoding the Independient Variable
transformX = ColumnTransformer([("Servers", OneHotEncoder(), [1,2,3])], remainder="passthrough")
X = transformX.fit_transform(X)


#Encoding the Dependent Variable
transformY= ColumnTransformer([("Attacks", OneHotEncoder(), [0])], remainder="passthrough")
Y = transformY.fit_transform(Y)
Y = pd.DataFrame(Y)

#Encoding Categorical Data for Test Set
#Encoding the Independient Variable
transformA = ColumnTransformer([("Servers", OneHotEncoder(), [1,2,3])], remainder="passthrough")
A = transformA.fit_transform(A)
    
#Encoding the Dependent Variable
transformZ = ColumnTransformer([("Attacks", OneHotEncoder(), [0])], remainder="passthrough")
Z = transformZ.fit_transform(Z)

###############################################################################################################
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)
#pd.get_dummies(,drop_first=True)
###############################################################################################################


#Because we are using numerical-value-only clustering techniques to analyze the NSL-KDD dataset,
#we need to normalize the values in the dataset, as Ibrahim., et. al. describe (page 112).
#We complete the normalization process below:
from sklearn.preprocessing import Normalizer

normalizer = Normalizer().fit(X)
X = normalizer.transform(X)

normalizer = Normalizer().fit(A)
A = normalizer.transform(A)


trainData = np.array(X)
trainLabel = np.array(Y)

testData =  np.array(A)
testLabel = np.array(Z)

###############################################################################################################
#model = LogisticRegression(solver = 'lbfgs')
#model.fit(trainData,trainLabel)
###############################################################################################################


#Elbow Method
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
#KMeans
#Applying K-mea(n_clusters = 5)
KMEANS = KMeans(n_clusters = 4, init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
kmeans = KMEANS.fit(trainData)
kmeans.labels_
pd.crosstab(attacks,kmeans.labels_)


#DBSCAN
from sklearn.cluster import DBSCAN
# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.2, min_samples = 1000).fit(trainData)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
# Analyzing Results of DBSCAN by Crosstab
crostab = pd.crosstab(attacks,labels)


# F-Score implementation
from sklearn.metrics import f1_score
f1 = f1_score(Y,kmeans.labels_, average="weighted") #[None, 'micro', 'macro', 'weighted']
f1
