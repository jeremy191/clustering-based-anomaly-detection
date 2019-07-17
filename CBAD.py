#@authors:jeremyperez,bethanydanner

#reset -f


import numpy as np
import pandas as pd 
import os
import warnings
#import matplotlib.pyplot as plt
clear = lambda:os.system('clear')
#warnings.filterwarnings('ignore')

def getDataSet():
    while True:
        print("**************************************************")
        print("DATA SET MENU")
        print("**************************************************")
        print("1.NSL-KDD")
        print("2.IDS 2017")
        
        option = input("Option:")
        
        if option == "1" or option == "2":
            break
    
    path = input("Path of the File:")
    print("\n\nReading Dataset...")
    
    return path,option

def readingData(path,dataSetOption):
    #Reading the Train Dataset

        
    if dataSetOption == "2":#Checking if data set has header
        dataSet = pd.read_csv(path,low_memory=False)
    
    elif dataSetOption == "1":
        dataSet = pd.read_csv(path, header = None,low_memory=False)
        
    
    
    return dataSet


#Getting The data we want to test for the clustering algorithms
def gettingVariables(dataSet,datasetOption):

    if datasetOption == "1":
        while True:
            print("Variables Menu\n")
            print("1.Data set with Categorical data")
            print("2.Data set without Categorical data")
            print("3.Data set without protocols to start  using risk Values\n")
            option = input("Enter option :")
            
            
            if option == "1" or option == "2" or option == "3":
                break
            else:
                
                print("Error\n\n")
            
        
        if option == "1":
            #Getting the Dependent and independent Variables
            X = dataSet.iloc[:,:-2].values # Data, Get all the rows and all the clums except all the colums - 2
            Y = dataSet.iloc[:,42].values# Labels
            return X,Y,option
        
        elif option == "2":
            #Removing Categorical data from the data set
            X = dataSet.iloc[:,[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]].values
            X = pd.DataFrame(X)
            Y = dataSet.iloc[:,42].values# Labels
            
            return X,Y,option
        
        elif option == "3":
            #Removing Protocols to start using risk Values
            X = dataSet.iloc[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]].values
            Y = dataSet.iloc[:,42].values# Labels
            
            return X,Y,option
    
    elif datasetOption == "2":
    #############################################################################
    #CHECK IF  MISSING DATA
    ############################################################################# 
        isMissing = str(dataSet.isnull().values.any()) #Using String instead of Boolean because ("cannot unpack non-iterable numpy.bool object")
       
        if isMissing == 'True':
            #if data set has infinity values replace them with none
            dataSet = dataSet.replace('Infinity', np.nan) #Replacing Infinity values with nan values
           
            missingValIndex = []
            total = dataSet.isnull().sum().sum()
            
            for rows in dataSet:
                    
                if dataSet[rows].isnull().sum() != 0:
                    missingValIndex.append(rows)
            print("**************************************************")
            print("Data has missing values")
            print("**************************************************")
            print("Rows with missing values:",missingValIndex)
            print("Total missing Values -> " , total)
    #############################################################################
    #END OF CHECK IF  MISSING DATA
    ############################################################################# 
            
    #############################################################################
    #MANAGE MISSING DATA
    #############################################################################        
        while True:
            print("\n**************************************************")
            print("Manage Missing Values ")
            print("**************************************************")
            print("1.Eliminate Catg. w/ Missing Values")
            print("2.Impute 0 for Missing Values")
            print("3.Impute Mean for Missing Values")
            print("4.Impute Median for Missing Values")
            print("5.Impute Mode for Missing Values")
            missingDataOption = input("Option:")
    
            if missingDataOption == "1" or missingDataOption == "2" or missingDataOption == "3" or missingDataOption == "4" or missingDataOption == "5":
                break
    
    
        if missingDataOption == "1":
            for row in missingValIndex:
                del dataSet[row]
        
            print("#########################################################################")
            print("Columns Succesfully Removed")
            print("#########################################################################")
    
        elif missingDataOption == "2":
            for row in missingValIndex:
                dataSet[row] = dataSet[row].fillna(0)
        
            print("#########################################################################")
            print("Sucessfully Filled Missing Values with 0")
            print("#########################################################################")
    
    
        elif missingDataOption == "3":
            for row in missingValIndex:
                dataSet[row] = dataSet[row].astype(float)
                dataSet[row] = dataSet[row].fillna(dataSet[row].mean())
        
            print("#########################################################################")
            print("Sucessfully Filled Missing Values with Mean")
            print("#########################################################################")
    
        elif missingDataOption == "4":
            for row in missingValIndex:
                median = dataSet[row].median()
                dataSet[row].fillna(median, inplace=True)
            print("#########################################################################")
            print("Sucessfully Filled Missing Values with Median")
            print("#########################################################################")
    
        elif missingDataOption == "5":
    
            for row in missingValIndex:
                dataSet[row] = dataSet[row].fillna(dataSet[row].mode()[0])
    
            print("#########################################################################")
            print("Sucessfully Filled Missing Values with Mode ")
            print("#########################################################################")
       
#############################################################################
#END OF MISSING DATA
#############################################################################
        if missingDataOption == "1":#Check if missing data was removed
            X = dataSet.iloc[:,:-1].values
            Y = dataSet.iloc[:,76].values#Labels
            option = "None"
        else: #If it was another Option that do not remove data rows
        
            X = dataSet.iloc[:,:-1].values
            Y = dataSet.iloc[:,78].values#Labels
            option = "None" #This data does not have categorical features so dataOption is none
        
        return X,Y,option
    




    
def encodingLabels(labels,dataOption,datasetOption):
    
    if datasetOption == "1": #Check if the data set choosen is NSL-KDD or IDS2017
        
        if dataOption == "1" or dataOption == "2" or dataOption == "3":
            
            while True:
                print("Encoding Menu\n")
                print("1.Binary Encode true labels - > 'normal = 0','abnormal = 1'")
                print("2.Five main categories encode true labels -> normal = 0,DoS = 1,Probe = 2,R2L = 3,U2R = 4'")
                encodeOption = input("Enter option :") 
    
                if encodeOption == "1" or encodeOption == "2":
                    break
                else:
                    
                    print("Error\n\n")
    
    
            if encodeOption == "1":
                #Binary Categories
                attackType  = {'normal':"normal", 'neptune':"abnormal", 'warezclient':"abnormal", 'ipsweep':"abnormal",'back':"abnormal", 'smurf':"abnormal", 'rootkit':"abnormal",'satan':"abnormal", 'guess_passwd':"abnormal",'portsweep':"abnormal",'teardrop':"abnormal",'nmap':"abnormal",'pod':"abnormal",'ftp_write':"abnormal",'multihop':"abnormal",'buffer_overflow':"abnormal",'imap':"abnormal",'warezmaster':"abnormal",'phf':"abnormal",'land':"abnormal",'loadmodule':"abnormal",'spy':"abnormal",'perl':"abnormal"} 
                attackEncodingCluster  = {'normal':0,'abnormal':1}
    
                labels[:] = [attackType[item] for item in labels[:]] #Encoding the binary data
                labels[:] = [attackEncodingCluster[item] for item in labels[:]]#Changing the names of the labels to binary labels normal and abnormal
                return labels,encodeOption
    
            elif encodeOption == "2":
                #4 Main Categories
                #normal = 0
                #DoS = 1
                #Probe = 2
                #R2L = 3
                #U2R = 4
    
                attackType  = {'normal':"normal", 'neptune':"DoS", 'warezclient':"R2L", 'ipsweep':"Probe",'back':"DoS", 'smurf':"DoS", 'rootkit':"U2R",'satan':"Probe", 'guess_passwd':"R2L",'portsweep':"Probe",'teardrop':"DoS",'nmap':"Probe",'pod':"DoS",'ftp_write':"R2L",'multihop':"R2L",'buffer_overflow':"U2R",'imap':"R2L",'warezmaster':"R2L",'phf':"R2L",'land':"DoS",'loadmodule':"U2R",'spy':"R2L",'perl':"U2R"} 
                attackEncodingCluster  = {'normal':0,'DoS':1,'Probe':2,'R2L':3, 'U2R':4} #Main Categories
    
                labels[:] = [attackType[item] for item in labels[:]] #Encoding the main 4 categories
                labels[:] = [attackEncodingCluster[item] for item in labels[:]]# Changing the names of attacks into 4 main categories
                return labels,encodeOption
        else:
            return labels
    
    
    elif datasetOption == "2":#Check if the data set choosen is NSL-KDD or IDS2017
        labels = np.array(labels,dtype= object)
        attackEncoding  = {'BENIGN': 0,'DoS slowloris': 1,'DoS Slowhttptest': 2,'DoS Hulk': 3, 'DoS GoldenEye': 4, 'Heartbleed': 5} #Main Categories
        labels[:] = [attackEncoding[item] for item in labels[:]]# Changing the names of attacks into 4 main categories
    
        return labels



#Encoding the data using one hot encoding and using Main attacks categories or binary categories
def oneHotEncodingData(data,dataOption):
        
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    #We use One hot encoding to pervent the machine learning to atribute the categorical data in order. 
    #What one hot encoding(ColumnTransformer) does is, it takes a column which has categorical data, 
    #which has been label encoded, and then splits the column into multiple columns.
    #The numbers are replaced by 1s and 0s, depending on which column has what value
    #We don't need to do a label encoded step because ColumnTransformer do one hot encode and label encode!
    #Encoding the Independient Variable
    if dataOption == "1": #Only for dataset with Categorical Data
        transform = ColumnTransformer([("Servers", OneHotEncoder(categories = "auto"), [1,2,3])], remainder="passthrough")
        data = transform.fit_transform(data)
        print("#########################################################################")
        print("Data has been Successfully One Hot Encoded")
        print("#########################################################################\n\n")

        return data
        
    else:
        return data #return data with no changes


def riskEncodingData(data,labels,dataOption):#This function is only for risk testing only
    #Manually Encoding for the attacks types only
    if dataOption == "3": #if data option is risk Value
        data = pd.DataFrame(data)
        servers  = {'http':0.01, 'domain_u':0, 'sunrpc':1, 'smtp':0.01, 'ecr_i':0.87, 'iso_tsap':1, 'private':0.97, 'finger':0.27, 'ftp':0.26, 'telnet':0.48,'other':0.12,'discard':1, 'courier':1, 'pop_3':0.53, 'ldap':1, 'eco_i':0.8, 'ftp_data':0.06, 'klogin':1, 'auth':0.31, 'mtp':1, 'name':1, 'netbios_ns':1,'remote_job':1,'supdup':1,'uucp_path':1,'Z39_50':1,'csnet_ns':1,'uucp':1,'netbios_dgm':1,'urp_i':0,'domain':0.96,'bgp':1,'gopher':1,'vmnet':1,'systat':1,'http_443':1,'efs':1,'whois':1,'imap4':1,'echo':1,'link':1,'login':1,'kshell':1,'sql_net':1,'time':0.88,'hostnames':1,'exec':1,'ntp_u':0,'nntp':1,'ctf':1,'ssh':1,'daytime':1,'shell':1,'netstat':1,'nnsp':1,'IRC':0,'pop_2':1,'printer':1,'tim_i':0.33,'pm_dump':1,'red_i':0,'netbios_ssn':1,'rje':1,'X11':0.04,'urh_i':0,'http_8001':1,'aol':1,'http_2784':1,'tftp_u':0,'harvest':1}
        data[1] = [servers[item] for item in data[1]]

        servers_Error  = {'REJ':0.519, 'SF':0.016, 'S0':0.998, 'RSTR':0.882, 'RSTO':0.886,'SH':0.993,'S1':0.008,'RSTOS0':1,'S3':0.08,'S2':0.05,'OTH':0.729} 
        data[2] = [servers_Error[item] for item in data[2]]

        print("#########################################################################")
        print("Data has ben risk Encoded Successfully")
        print("#########################################################################\n\n")

        return data,labels
        
    else:
        
        return data,labels #return data with no changes
            
    


def normalizing(data):#Scalign the data with the normalize method
    
    from sklearn.preprocessing import Normalizer
    #Because we are using numerical-value-only clustering techniques to analyze the NSL-KDD dataset,
    #we need to normalize the values in the dataset, as Ibrahim., et. al. describe (page 112).
    #Normalize works by scaling the features in a range of [0,1]
    #We complete the normalization process below:
    normalizer = Normalizer().fit(data)
    data = normalizer.transform(data)   
    print("#########################################################################")
    print("Data has ben Successfully normalized")
    print("#########################################################################\n\n")

    return data



    
def shuffleData(X):
    while True:
        option = input("Suffle Data [y]/[n] :")
        
        if option == "y" or option == "n":
            break
        else:
            
            print("Error\n\n")
    
    if option == "y":
        
        np.random.shuffle(X)
        print("#########################################################################")
        print("Data has been Successfully Shuffled")
        print("#########################################################################\n\n")
        return X
    else:
        print("\n\n")
        return X





def kmeansClustering(data,labels):#K-means algorithm 
    from sklearn.cluster import KMeans

    while True:
        print("#########################################################################")
        print("KMEANS ALGORITHM")
        print("#########################################################################\n\n")
              
        nClusters = input("Numbers of clusters:")
        
        try:
            nClusters = int(nClusters)
            
        except ValueError:
            
            print("Error\n\n")
            
        if type(nClusters) == int:
            n = 0
            clusterArray = []
            
            while n < nClusters:#Converting nCluster into an array of n clusters [n]
                clusterArray.append(n)
                n+=1
            break
        
    while True:
        init = input("Initialization method [k-means++,random]:")
        
        if init == "k-means++" or init == "random":
            
            break
    
    print("\nClustering...\n")
    
    
    KMEANS = KMeans(n_clusters = nClusters, init = init,max_iter = 300,n_init = 10,random_state = 0)
    kmeans = KMEANS.fit(data)
    klabels = kmeans.labels_
    
    #Kmeans Results
    kmeansR = pd.crosstab(labels,klabels)
    maxV = kmeansR.idxmax()
    return klabels,clusterArray,kmeansR,maxV,




def kF1(klabels,labels,maxKvalue,nClusters):#F1 Score for Kmeans
    from sklearn.metrics import f1_score
    #Encoding data to F-score
    
    
    n = 0 # counter
    dictionaryCluster  = {} # creating an empty dictionary 
    
    
    while n < len(nClusters):# while counter < number of clusters
        dictionaryCluster[nClusters[n]] = maxKvalue[n] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        
    klabels[:] = [dictionaryCluster[item] for item in klabels[:]] # match key with the index of klabels and replace it with key value
            
    labels = np.array(labels,dtype = int) # Converting labels into a int array
    
    while True:
        
        average = input("Average Method[weighted,None,micro,macro]:")
        
        if average == "weighted" or average == "micro" or average == "macro" or average == "None":
            break
        
    f1 = f1_score(labels,klabels, average = average,labels=np.unique(klabels))
    
    return f1




def kAUC(klabels,labels,maxKvalue,nClusters):
    from sklearn.metrics import roc_auc_score

    n = 0 # counter
    dictionaryCluster  = {} # creating an empty dictionary 
    
    while n < len(nClusters):# while counter < number of clusters
        dictionaryCluster[nClusters[n]] = maxKvalue[n] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        
    klabels[:] = [dictionaryCluster[item] for item in klabels[:]] # match key with the index of klabels and replace it with key value
    
    
    labels = np.array(labels,dtype = int) #Making sure that labels are in a int array
    
    while True:
        
        average = input("Average Method[None,micro,macro,weighted]:")
        
        if average == "None" or average == "micro" or average == "macro" or average == "weighted":
            break
        
    AUC = roc_auc_score(labels, klabels, average = average)
    return AUC





def kMNI(klabels,labels,maxKvalue,nClusters):
    from sklearn.metrics import normalized_mutual_info_score
    
    n = 0 # counter
    dictionaryCluster  = {} # creating an empty dictionary 
    
    while n < len(nClusters):# while counter < number of clusters
        dictionaryCluster[nClusters[n]] = maxKvalue[n] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        
    klabels[:] = [dictionaryCluster[item] for item in klabels[:]] # match key with the index of klabels and replace it with key value
    
    labels = np.array(labels,dtype = int) #Making sure that labels are in a int array
    
    while True:
        
        average = input("Average Method[geometric,min,arithmetic,max]:")
        
        if average == "geometric" or average == "min" or average == "arithmetic" or average == "max":
            break
    
    MNI = normalized_mutual_info_score(labels, klabels, average_method = average)
    
    return MNI



def kARS(klabels,labels,maxKvalue,nClusters):
    from sklearn.metrics import adjusted_rand_score
    
    n = 0 # counter
    dictionaryCluster  = {} # creating an empty dictionary 
    
    while n < len(nClusters):# while counter < number of clusters
        dictionaryCluster[nClusters[n]] = maxKvalue[n] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        
    klabels[:] = [dictionaryCluster[item] for item in klabels[:]] # match key with the index of klabels and replace it with key value
    
    labels = np.array(labels,dtype = int) #Making sure that labels are in a int array
    
    ars = adjusted_rand_score(labels, klabels)
    
    return ars



def dbscanClustering(data,labels):#DBSCAN algorithm
    from sklearn.cluster import DBSCAN
    
    while True:
        
        print("#########################################################################")
        print("DBSCAN ALGORITHM")
        print("#########################################################################\n\n")
              
        epsilon = input("epsilon[Decimal]:")
        
        try:
            epsilon = float(epsilon)
            
        except ValueError:
            
            print("Enter a Decimal number")
            
            
        if type(epsilon) == float:
            break
        
    while True:
        minSamples = input("Min Samples[Integer]:")
        
        try:
            minSamples = int(minSamples)
            
        except ValueError:
            
            print("Enter a Integer Number")
            
        if type(minSamples) == int:
            break
        
    while True:
        algorithm = input("Algorithm['auto’, ‘ball_tree’, ‘kd_tree’, 'brute']:")
            
        if algorithm == "auto" or algorithm == "ball_tree" or algorithm == "kd_tree" or algorithm == "brute":
            break
        else:
            
            print("Error\n\n")
            
    
    print("\nClustering...\n")
    
    #Compute DBSCAN
    db = DBSCAN(eps= epsilon, min_samples = minSamples,algorithm = algorithm).fit(data) #{auto, ball_tree, kd_tree, brute}
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    dblabels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(dblabels))
    n_noise_ = list(dblabels).count(-1)
    
    n = -1  # DBSCAN return index -1 cluster
    dbClusterArray = []
    while n + 1 < n_clusters:
        dbClusterArray.append(n)
        n += 1
    
    #DBSCAN Results
    dbscanR = pd.crosstab(labels,dblabels)
    maxValue = dbscanR.idxmax()

    return dblabels,dbClusterArray,n_noise_,dbscanR,maxValue




def dbF1(dblabels,labels,dbClusters,maxDBvalue):#F1 score for DBSCAN
    from sklearn.metrics import f1_score
    #Encoding data to F-score
    
    
    n = 0 # counter
    c = -1 # - counter max Value has negative index
    dictionaryCluster  = {} # creating an empty dictionary 
    
    while n < len(dbClusters):# while counter < number of clusters
        dictionaryCluster[dbClusters[n]] = maxDBvalue[c] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        c+=1
    
        
    dblabels[:] = [dictionaryCluster[item] for item in dblabels[:]] # match key with the index of klabels and replace it with key value
    
    labels = np.array(labels,dtype = int) #Making sure that labels are in a int array
    while True:
        
        average = input("Average Method[weighted,None,micro,macro]:")
        
        if average == "weighted" or average == "micro" or average == "macro" or average == "None":
            break
        
        else:
            
            print("Error\n\n")
    
    f1 = f1_score(labels,dblabels, average = average,labels=np.unique(dblabels))
    
    return f1




def dbAUC(dblabels,labels,dbClusters,maxDBvalue):
    from sklearn.metrics import roc_auc_score
    
    n = 0 # counter
    c = -1 # - counter max Value has negative index
    dictionaryCluster  = {} # creating an empty dictionary 
    
    while n < len(dbClusters):# while counter < number of clusters
        dictionaryCluster[dbClusters[n]] = maxDBvalue[c] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        c+=1
        
    dblabels[:] = [dictionaryCluster[item] for item in dblabels[:]] # match key with the index of klabels and replace it with key value
    
    labels = np.array(labels,dtype = int)#Making sure that labels are in a int array
    
    while True:
        
        average = input("Average Method[None,micro,macro,weighted]:")
        
        if average == "None" or average == "micro" or average == "macro" or average == "weighted":
            break
        
        else:
            
            print("Error\n\n")
        
    AUC = roc_auc_score(labels, dblabels, average = 'weighted')
    
    return AUC




def dbMNI(dblabels,labels,dbClusters,maxDBvalue):
    from sklearn.metrics import normalized_mutual_info_score
    
    n = 0 # counter
    c = -1 # - counter max Value has negative index
    dictionaryCluster  = {} # creating an empty dictionary 
    
    while n < len(dbClusters):# while counter < number of clusters
        dictionaryCluster[dbClusters[n]] = maxDBvalue[c] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        c+=1
    
    
    labels = np.array(labels,dtype = int) #Making sure that labels are in a int array


    while True:
        
        average = input("Average Method[geometric,min,arithmetic,max]:")
        
        if average == "geometric" or average == "min" or average == "arithmetic" or average == "max":
            break
        else:
            
            print("Error\n\n")
        
    MNI = normalized_mutual_info_score(labels, dblabels, average_method='min')
    
    return MNI



def dbARS(dblabels,labels,dbClusters,maxDBvalue):
    from sklearn.metrics import adjusted_rand_score
    
    n = 0 # counter
    c = -1 # - counter max Value has negative index
    dictionaryCluster  = {} # creating an empty dictionary 
    
    while n < len(dbClusters):# while counter < number of clusters
        dictionaryCluster[dbClusters[n]] = maxDBvalue[c] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        c+=1
    
    ars = adjusted_rand_score(labels, dblabels,dbClusters,maxDBvalue)
    
    return ars


#def isolationForest(data,labels):
#    from sklearn.ensemble import IsolationForest
#    
#    isolationForest = IsolationForest(n_estimators = 500,max_samples = "auto",behaviour = "new",contamination = "auto")
#    ifLabels = isolationForest.fit_predict(data)
#    
#    
#    ifLabels = np.array(ifLabels,dtype = object)
#
#    ifR = pd.crosstab(labels,ifLabels)
#    
#    MaxIfVal = ifR.idxmax()
#    MaxIfVal.reset_index(drop=True)
#    
#    nClusters = len(ifR)
#    
#    
#    return ifLabels,ifR,MaxIfVal,nClusters

#def ifF1(ifLabels,labels,dbClusters,MaxIfVal,nClusters):
#    from sklearn.metrics import f1_score
#    #Encoding data to F-score
    


#def LOF(data):
#    from sklearn.neighbors import LocalOutlierFactor 
#    LOF = LocalOutlierFactor(contamination = "auto")
#    lof = LOF.fit(data)
#    lofOutlier = lof.negative_outlier_factor_
#    
#    return lofOutlier
##########################################################################
#lof = LOF(data)
#lofR = pd.crosstab(labels,lof)
#lofR.idxmax()
##########################################################################
#
#
#
#def lofF1(lof,labels):
#    from sklearn.metrics import f1_score
#    i = 0
#    for item in lof:
#        if item >= -1.5:
#            lof[i] = 0
#        elif item < -1.5:
#           lof[i] = 1
#        i+= 1
#    labels = np.array(labels,dtype = int)
#    lof = np.array(lof,dtype = int)
#    f1 = f1_score(labels,lof, average = 'weighted') #[None, 'micro', 'macro', 'weighted']
#    return f1
###########################################################################
#lofF1 = lofF1(lof,labels)
#lofF1
###########################################################################




def main():
    ##########################################################################
    path,dataSetOption = getDataSet()
    #########################################################################
    #########################################################################
    #/Users/jeremyperez/Desktop/CICIDS2017.csv
    #/Users/jeremyperez/Jupyter/NSL-KDD/KDDTrain+.csv
    #/Users/bethanydanner/Google_Drive/documents/python_code/codeLines_newData/CICIDS2017.csv
    #/Users/bethanydanner/Google_Drive/documents/python_code/clustering-based-anomaly-detection/Dataset/NSL-KDD/KDDTrain+.csv", header = None)
    dataSet = readingData(path,dataSetOption)
    
    #########################################################################
    #########################################################################
    data,labels,dataOption = gettingVariables(dataSet,dataSetOption) #Getting the Data we want to use for the algorithms
    #########################################################################
    #########################################################################
    try:
        labels,encodeOption = encodingLabels(labels,dataOption,dataSetOption) #Encoding the true labels
    except ValueError:
        labels = encodingLabels(labels,dataOption,dataSetOption) #Encoding the true labels
        
    
    #########################################################################
    #########################################################################
    data = shuffleData(data)
    #########################################################################
    #########################################################################
    data = oneHotEncodingData(data,dataOption) #One hot Encode with the complete data
    #########################################################################
    #########################################################################
    data,labels = riskEncodingData(data,labels,dataOption)
    #########################################################################
    #########################################################################
    data = normalizing(data)
    #########################################################################
    
    while True:  
        while True:
            print("#########################################################################")
            print("Algorithm Menu")
            print("#########################################################################\n\n")
            
            print("1.Kmeans")
            print("2.Dbscan")
            print("3.Isolation Forest")
            print("4.Local Factor Outlier")
            
            algrithmOption = input("option:")
            
            if algrithmOption == "1" or algrithmOption == "2" or algrithmOption == "3" or algrithmOption == "4":
                    break
            else:
                
                print("Error\n\n")
    
        
        if algrithmOption == "1":
            #########################################################################
            #KMEANS
            
            klabels,kClusters,kmeansR,maxKvalue = kmeansClustering(data,labels)
            
            print("#########################################################################")
            print("KMEANS RESULTS\n\n")
            print("Clusters -> ",kClusters,"\n")
            print(kmeansR,"\n\n")
            print("Max True Label","\n\n",maxKvalue)
            print("\n#########################################################################\n\n\n")

            #########################################################################
            
            
            while True:
                print("#########################################################################")
                print("Kmeans Score Metrics Menu")
                print("#########################################################################\n\n")
            
                print("1.F1 Score")
                print("2.AUC")
                print("3.Normalized Mutual Info Score")
                print("4.Adjusted Rand Score")
            
                kScoreOption = input("option:")
                
                if kScoreOption == "1" or kScoreOption == "2" or kScoreOption == "3" or kScoreOption == "4":
                    break
                else:
                    
                    print("Error\n\n")
        
            
            
            
            if kScoreOption == "1":
                    #########################################################################
                    #F1 Score
                    kmeansF1 = kF1(klabels,labels,maxKvalue,kClusters)
                    print("#########################################################################")
                    print("KMEANS F1 Score -> ",kmeansF1)
                    print("#########################################################################")
                    #########################################################################
            
            elif kScoreOption == "2":
                    #########################################################################
                    try:
                        kmeansAUC = kAUC(klabels,labels,maxKvalue,kClusters)
                        print("#########################################################################")
                        print("AUC Score -> ",kmeansAUC)
                        print("#########################################################################")
                    except Exception as Error:
                        print("\n*************************************************************************")
                        print(Error)
                        print("*************************************************************************")

                    #########################################################################
            
            elif kScoreOption == "3":
                    #########################################################################
                    kmeansMNI = kMNI(klabels,labels,maxKvalue,kClusters)
                    print("#########################################################################")
                    print("KMEANS Normalized Mutual Info Score -> ",kmeansMNI)
                    print("#########################################################################")
                    #########################################################################
        
            elif kScoreOption == "4":
                
                #########################################################################
                kmeansARS = kARS(klabels,labels,maxKvalue,kClusters)
                print("#########################################################################")
                print("KMEANS Adjusted Rand Score -> ",kmeansARS)
                print("#########################################################################")
                #########################################################################
            
            
        elif algrithmOption == "2":
            #########################################################################
            #DBSCAN
            dblabels,dbClusters,nNoises,dbscanR,maxDBvalue = dbscanClustering(data,labels) 
            
            print("#########################################################################")
            print("DBSCAN RESULTS\n\n")
            print("Clusters -> ",dbClusters,"\n")
            print(dbscanR,"\n\n")
            print("Max True Label","\n\n",maxDBvalue)
            print("\n#########################################################################\n\n\n")
            #########################################################################
            
            
            print("#########################################################################")
            print("Dscan Score Metrics Menu")
            print("#########################################################################\n\n")
            
            print("1.F1 Score")
            print("2.AUC")
            print("3.Normalized Mutual Info Score")
            print("4.Adjusted Rand Score")
            
            while True:
                
                dbScoreOption = input("option:")
                
                if dbScoreOption == "1" or dbScoreOption == "2" or dbScoreOption == "3" or dbScoreOption == "4":
                    break
                else:
                    
                    print("Error\n\n")
        
            if dbScoreOption == "1":
                #########################################################################
                #F1 Score dbscan
                dbscanF1 = dbF1(dblabels,labels,dbClusters,maxDBvalue)
                print("#########################################################################")
                print("DBSCAN F1 Score -> ",dbscanF1)
                print("#########################################################################")
                #########################################################################
                
            elif dbScoreOption == "2":
                #########################################################################
                try:
                    dbscanAUC = dbAUC(dblabels,labels,dbClusters,maxDBvalue)
                    print("#########################################################################")
                    print("DBSCAN AUC Score -> ",dbscanAUC)
                    print("#########################################################################")
                except Exception as Error:
                    print("\n*************************************************************************")
                    print(Error)
                    print("*************************************************************************")
                    
                
                #########################################################################
            
            elif dbScoreOption == "3":
                #########################################################################
                dbscanMNI = dbMNI(dblabels,labels,dbClusters,maxDBvalue)
                print("#########################################################################")
                print("DBSCAN Normalized Mutual Info Score -> ",dbscanMNI)
                print("#########################################################################")
                #########################################################################
            
            elif dbScoreOption == "4":
                #########################################################################
                dbscanARS = dbARS(dblabels,labels)
                print("#########################################################################")
                print("DBSCAN Adjusted Rand Score -> ",dbscanARS)
                print("#########################################################################")
                #########################################################################
                
            
        elif algrithmOption == "3":
            #########################################################################
            #ifLabels,ifR,MaxIfVal,ifNclusters = isolationForest(data,labels)
            #########################################################################
            print("x")
    
        elif algrithmOption == "4":
            print("x")
        
        while True:
            
            decision = input("You want to another Test[y/n]:")
            
            if decision == "y" or  decision == "n":
                break
            else:
                
                print("Error\n\n")
        
        
        if decision == "n":
            break
        
        
      
#*************
#    MAIN
#*************
main()