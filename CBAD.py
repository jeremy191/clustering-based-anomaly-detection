#@authors:jeremyperez,bethanydanner

#reset -f


import numpy as np
import pandas as pd 
import time
import os

#import matplotlib.pyplot as plt
clear = lambda:os.system('clear')

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
    print("\nReading Dataset...")
    
    return path,option

def readingData(path): #Reading the Train Dataset
    
    while True:
        
        option = input("Dataset has feature names[y/n]:") 
        
        if option == "y" or option == "n":
            break
            
        print("\nReading Dataset...") 
        
    if option == "y":
        dataSet = pd.read_csv(path,low_memory=False)
    
    elif option == "n":
        dataSet = pd.read_csv(path, header = None,low_memory=False)
            
    return dataSet


def checkMissing(X):    
    isMissing = str(X.isnull().values.any()) #Using String instead of Boolean because ("cannot unpack non-iterable numpy.bool object")
    
    if isMissing == "True":
        #if data set has infinity values replace them with none
        X = X.replace('Infinity', np.nan) #Replacing Infinity values with nan values
           
        missingValIndex = []
        total = X.isnull().sum().sum()
        percent = (total / (X.count().sum() + X.isnull().sum().sum())) * 100
            
        for rows in X:
                    
            if X[rows].isnull().sum() != 0:
                missingValIndex.append(rows)
        print("\n\n**************************************************")
        print("Data has missing values")
        print("**************************************************")
        print("Features with missing values:",missingValIndex)
        print("Total missing Values -> " , total)
        print(percent,"%")
        
        return X
    
    else:
        
        return X


#Getting The data we want to test for the clustering algorithms
def gettingVariables(dataSet,dataSetOption):
   
    if dataSetOption == "1":
        while True:
            print("\n\n**************************************************")
            print("Variables Menu")
            print("**************************************************")
            print("1.Data set with categorical data oneHot encoded")
            print("2.Data set with categorical data removed")
            print("3.Data set with Risk Values replacing Server Type and Flag Features; Protocol Data oneHot encoded")
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
            Y = dataSet.iloc[:,42].values# Labels
            
            return X,Y,option
        
        elif option == "3":
            #Risk Encode Categorical features
            X = dataSet.iloc[:,:-2].values
            Y = dataSet.iloc[:,42].values# Labels
            
            return X,Y,option
    

    elif dataSetOption == "2":
        #############################################################################
        #GETTING VARIABLES
        #############################################################################
        missingValIndex = []
        for rows in dataSet: #Getting features index with missing values
            if dataSet[rows].isnull().sum() != 0:
                    missingValIndex.append(dataSet)
                
        X = dataSet.iloc[:,:-1].values#data
        X = pd.DataFrame(X)
        Y = dataSet.iloc[:,78].values#Labels
        
        #############################################################################
        #Variables Got 
        #############################################################################
        
    #############################################################################
    #MANAGE MISSING DATA
    #############################################################################        
        while True:
            print("\n\n**************************************************")
            print("Manage Missing Values ")
            print("**************************************************")
            print("1.Eliminate Catg. w/ Missing Values")
            print("2.Impute 0 for Missing Values")
            print("3.Impute Mean for Missing Values")
            print("4.Impute Median for Missing Values")
            print("5.Impute Mode for Missing Values")
            print("6.MICE Method")
            missingDataOption = input("Option:")
    
            if missingDataOption == "1" or missingDataOption == "2" or missingDataOption == "3" or missingDataOption == "4" or missingDataOption == "5" or missingDataOption == "6":
                break
    
    
        if missingDataOption == "1":
            deletedColumns = []
            numColumns = len(X.columns)
            for row in missingValIndex:
                deletedColumns.append(row)
                del X[row]
        
            print("#\n\n########################################################################")
            print("Columns Succesfully Removed")
            print(len(deletedColumns),"of",numColumns,"were deleted")
            print("Columns Names -> ",deletedColumns)
            print("#########################################################################")
    
        elif missingDataOption == "2":
            for row in missingValIndex:
                X[row] = X[row].fillna(0)
        
            print("\n\n#########################################################################")
            print("Sucessfully Filled Missing Values with 0")
            print("#########################################################################")
    
    
        elif missingDataOption == "3":
            for row in missingValIndex:
                X[row] = X[row].astype(float)
                X[row] = X[row].fillna(X[row].mean())
        
            print("\n\n#########################################################################")
            print("Sucessfully Filled Missing Values with Mean")
            print("#########################################################################")
    
        elif missingDataOption == "4":
            for row in missingValIndex:
                median = X[row].median()
                X[row].fillna(median, inplace=True)
            print("\n\n#########################################################################")
            print("Sucessfully Filled Missing Values with Median")
            print("#########################################################################")
    
        elif missingDataOption == "5":
    
            for row in missingValIndex:
                X[row] = X[row].fillna(X[row].mode()[0])
    
            print("\n\n#########################################################################")
            print("Sucessfully Filled Missing Values with Mode ")
            print("#########################################################################")
        
        elif missingDataOption == "6": 
            from sklearn.impute import SimpleImputer
            #"Imputation transformer for completing missing values."(Univariate)
            X = SimpleImputer(missing_values = np.nan, strategy='mean', fill_value=None, verbose=0, copy=True).fit_transform(X)          
            print("\n\n#########################################################################")
            print("Sucessfully Imputed Simple Imputer ")
            print("#########################################################################")
                  
                  
        option = "None" #This data does not have categorical features so dataOption is none      
        return X,Y,option
       
#############################################################################
#END OF MISSING DATA
#############################################################################
    




    
def encodingLabels(Y,dataOption,datasetOption):
    
    if datasetOption == "1": #Check if the data set choosen is NSL-KDD or IDS2017
        
        if dataOption == "1" or dataOption == "2" or dataOption == "3":
            
            while True:
                print("\n\n#########################################################################")
                print("Encoding Menu")
                print("#########################################################################")
                print("1.Binary true labels: normal = 0, abnormal = 1")
                print("2.Multiclass true labels: normal = 0, DoS = 1, Probe = 2, R2L = 3, U2R = 4")
                encodeOption = input("Enter option :") 
    
                if encodeOption == "1" or encodeOption == "2":
                    break
                else:
                    
                    print("Error\n\n")
    
    
            if encodeOption == "1":
                #Binary Categories
                attackType  = {'normal':"normal", 'neptune':"abnormal", 'warezclient':"abnormal", 'ipsweep':"abnormal",'back':"abnormal", 'smurf':"abnormal", 'rootkit':"abnormal",'satan':"abnormal", 'guess_passwd':"abnormal",'portsweep':"abnormal",'teardrop':"abnormal",'nmap':"abnormal",'pod':"abnormal",'ftp_write':"abnormal",'multihop':"abnormal",'buffer_overflow':"abnormal",'imap':"abnormal",'warezmaster':"abnormal",'phf':"abnormal",'land':"abnormal",'loadmodule':"abnormal",'spy':"abnormal",'perl':"abnormal"} 
                attackEncodingCluster  = {'normal':0,'abnormal':1}
    
                Y[:] = [attackType[item] for item in Y[:]] #Encoding the binary data
                Y[:] = [attackEncodingCluster[item] for item in Y[:]]#Changing the names of the labels to binary labels normal and abnormal
                return Y,encodeOption
    
            elif encodeOption == "2":
                #4 Main Categories
                #normal = 0
                #DoS = 1
                #Probe = 2
                #R2L = 3
                #U2R = 4
                attackType  = {'normal': 'normal', 'neptune':'DoS', 'warezclient': 'R2L', 'ipsweep': 'Probe','back': 'DoS', 'smurf': 'DoS', 'rootkit': 'U2R','satan': 'Probe', 'guess_passwd': 'R2L','portsweep': 'Probe','teardrop': 'DoS','nmap': 'Probe','pod': 'DoS','ftp_write': 'R2L','multihop': 'R2L','buffer_overflow': 'U2R','imap': 'R2L','warezmaster': 'R2L','phf': 'R2L','land': 'DoS','loadmodule': 'U2R','spy': 'R2L','perl': 'U2R'} 
                attackEncodingCluster  = {'normal':0,'DoS':1,'Probe':2,'R2L':3, 'U2R':4} #Main Categories
    
                Y[:] = [attackType[item] for item in Y[:]] #Encoding the main 4 categories
                Y[:] = [attackEncodingCluster[item] for item in Y[:]]# Changing the names of attacks into 4 main categories
                return Y,encodeOption
        else:
            return Y
    
    
    elif datasetOption == "2":#Check if the data set choosen is NSL-KDD or IDS2017
        print("\n\n#########################################################################")
        print("Encoding Menu")
        print("#########################################################################")
        print("1.Binary true labels: normal = 0, abnormal = 1")
        print("2. Multiclass true labels: BENIGN= 0, DoS slowloris= 1, DoS Slowhttptest= 2, DoS Hulk= 3, DoS GoldenEye= 4, Heartbleed= 5")
        encodeOption = input("Enter option :")

        if encodeOption == "1":
            Y = np.array(Y,dtype= object)
            attackEncoding  = {'BENIGN': 0,'DoS slowloris': 1,'DoS Slowhttptest': 2,'DoS Hulk': 3, 'DoS GoldenEye': 4, 'Heartbleed': 5} #Main Categories
            Y[:] = [attackEncoding[item] for item in Y[:]]# Changing the names of attacks into 4 main categories
    
            return Y,encodeOption
        
        elif encodeOption == "2":
            Y = np.array(Y,dtype= object)
            attackType  = {'BENIGN': 'normal','DoS slowloris': 'abnormal','DoS Slowhttptest': 'abnormal','DoS Hulk': 'abnormal', 'DoS GoldenEye': 'abnormal', 'Heartbleed': 'abnormal'} #Binary Categories
            attackEncoding = {'normal': 0, 'abnormal': 1}
            
            Y[:] = [attackType[item] for item in Y[:]]# Changing the names of attacks into binary categories
            Y[:] = [attackEncoding[item] for item in Y[:]]# Changing the names of attacks into binary categories
            return Y,encodeOption
        
        else:
            return Y




#Encoding the data using one hot encoding and using Main attacks categories or binary categories
def oneHotEncodingData(X,dataOption):
        
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
        X = transform.fit_transform(X)
        print("\n\n#########################################################################")
        print("Data has been successfully One Hot Encoded")
        print("#########################################################################")

        return X
    elif dataOption == "3": #Only for risk data
        transform = ColumnTransformer([("Servers", OneHotEncoder(categories = "auto"), [1])], remainder="passthrough")
        X = transform.fit_transform(X)
        print("\n\n#########################################################################")
        print("Data has been successfully One Hot Encoded")
        print("#########################################################################")
        return X
        
    else:
        return X #return data with no changes


def riskEncodingData(X,dataOption):#This function is only for risk testing only
    #Manually Encoding for the attacks types only
    if dataOption == "3": #if data option is risk Value
        X = pd.DataFrame(X)
        servers  = {'http':0.01, 'domain_u':0, 'sunrpc':1, 'smtp':0.01, 'ecr_i':0.87, 'iso_tsap':1, 'private':0.97, 'finger':0.27, 'ftp':0.26, 'telnet':0.48,'other':0.12,'discard':1, 'courier':1, 'pop_3':0.53, 'ldap':1, 'eco_i':0.8, 'ftp_data':0.06, 'klogin':1, 'auth':0.31, 'mtp':1, 'name':1, 'netbios_ns':1,'remote_job':1,'supdup':1,'uucp_path':1,'Z39_50':1,'csnet_ns':1,'uucp':1,'netbios_dgm':1,'urp_i':0,'domain':0.96,'bgp':1,'gopher':1,'vmnet':1,'systat':1,'http_443':1,'efs':1,'whois':1,'imap4':1,'echo':1,'link':1,'login':1,'kshell':1,'sql_net':1,'time':0.88,'hostnames':1,'exec':1,'ntp_u':0,'nntp':1,'ctf':1,'ssh':1,'daytime':1,'shell':1,'netstat':1,'nnsp':1,'IRC':0,'pop_2':1,'printer':1,'tim_i':0.33,'pm_dump':1,'red_i':0,'netbios_ssn':1,'rje':1,'X11':0.04,'urh_i':0,'http_8001':1,'aol':1,'http_2784':1,'tftp_u':0,'harvest':1}
        X[2] = [servers[item] for item in X[2]]

        servers_Error  = {'REJ':0.519, 'SF':0.016, 'S0':0.998, 'RSTR':0.882, 'RSTO':0.886,'SH':0.993,'S1':0.008,'RSTOS0':1,'S3':0.08,'S2':0.05,'OTH':0.729} 
        X[3] = [servers_Error[item] for item in X[3]]

        print("\n\n#########################################################################")
        print("Data has been successfully risk Encoded")
        print("#########################################################################")

        return X
        
    else:
        
        return X #return data with no changes
            
    


def scaling(X):#Scalign the data with the normalize method
    

    while True:
            
            decision = input("Scale data [y/n]:")
            
            if decision == "y" or  decision == "n":
                break
            else:
                
                print("Error\n\n")
    
    if decision == "y":
        
            from sklearn.preprocessing import MinMaxScaler
            #Transforms features by scaling each feature to a given range.
            X =  MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
            print("\n\n#########################################################################")
            print("Data has been successfully scaled.")
            print("#########################################################################")
            return X
        
    else:
        return X

    
def shuffleData(X):
    from sklearn.utils import shuffle
    while True:
        option = input("Shuffle data [y]/[n]:")
        
        if option == "y" or option == "n":
            break
        else:
            
            print("Error\n\n")
    
    if option == "y":
        
        X = pd.DataFrame(X)
        X = shuffle(X)
        X.reset_index(inplace=True,drop=True)
        X = np.array(X)
        
        print("\n\n#########################################################################")
        print("Data has been successfully shuffled.")
        print("#########################################################################")
        return X
    else:
        
        return X





def kmeansClustering(X,Y):#K-means algorithm 
    from sklearn.cluster import KMeans

    while True:
        print("\n\n#########################################################################")
        print("KMEANS ALGORITHM")
        print("#########################################################################")
              
        nClusters = input("Number of clusters:")
        
        try:
            nClusters = int(nClusters)
            
        except ValueError:
            
            print("Error\n\n")
            
        if type(nClusters) == int:
            n = 0
            clusters = []
            
            while n < nClusters:#Converting nCluster into an array of n clusters [n]
                clusters.append(n)
                n+=1
            break
        
    while True:
        init = input("Initialization method [k-means++,random]:")
        
        if init == "k-means++" or init == "random":
            break

    print("\nClustering...\n")
    
    start_time = time.time()
    KMEANS = KMeans(n_clusters = nClusters, init = init,max_iter = 300,n_init = 10,random_state = 0)
    print("\n\nRun Time ->","--- %s seconds ---" % (time.time() - start_time))
    print("Data Successfully Clustered")
    kmeans = KMEANS.fit(X)
    Z = kmeans.labels_
    inertia = KMEANS.inertia_
    #Kmeans Results
    kmeansR = pd.crosstab(Y,Z)
    maxVal = kmeansR.idxmax()
    
    return Z,clusters,kmeansR,maxVal,inertia




def kF1(Z,Y,maxVal,clusters):#F1 Score for Kmeans
    from sklearn.metrics import f1_score
    #Encoding data to F-score
    
    
    n = 0 # counter
    dictionaryCluster  = {} # creating an empty dictionary 
    f1 = 0 #f1score
    average = ''
    
    while n < len(clusters):# while counter < number of clusters
        dictionaryCluster[clusters[n]] = maxVal[n] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        
    Z[:] = [dictionaryCluster[item] for item in Z[:]] # match key with the index of klabels and replace it with key value
            
    Y = np.array(Y,dtype = int) # Converting labels into a int array
    
    while True:
        
        average = input("Average Method[weighted,micro,macro,binary]:")
        
        if average == "weighted" or average == "micro" or average == "macro" or average == 'binary':
            break
        
    f1 = f1_score(Y,Z, average = average) #Forget the labels that where not predicted and gives lables that were predicted at least once
    
    return f1,dictionaryCluster



def kNMI(Z,Y,maxVal,clusters):
    from sklearn.metrics import normalized_mutual_info_score
    
    n = 0 # counter
    dictionaryCluster  = {} # creating an empty dictionary 
    NMI = 0
    average = ''
    
    while n < len(clusters):# while counter < number of clusters
        dictionaryCluster[clusters[n]] = maxVal[n] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        
    Z[:] = [dictionaryCluster[item] for item in Z[:]] # match key with the index of klabels and replace it with key value
    
    Y = np.array(Y,dtype = int) #Making sure that labels are in a int array
    
    while True:
        
        average = input("Average Method[geometric,min,arithmetic,max]:")
        
        if average == "geometric" or average == "min" or average == "arithmetic" or average == "max":
            break
    
    NMI = normalized_mutual_info_score(Y, Z, average_method = average)
    
    return NMI,dictionaryCluster



def kARS(Z,Y,maxVal,clusters):
    from sklearn.metrics import adjusted_rand_score
    
    n = 0 # counter
    dictionaryCluster  = {} # creating an empty dictionary 
    ars = 0
    
    while n < len(clusters):# while counter < number of clusters
        dictionaryCluster[clusters[n]] = maxVal[n] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        
    Z[:] = [dictionaryCluster[item] for item in Z[:]] # match key with the index of klabels and replace it with key value
    
    Y = np.array(Y,dtype = int) #Making sure that labels are in a int array
    
    ars = adjusted_rand_score(Y, Z)
    
    return ars,dictionaryCluster



def dbscanClustering(X,Y):#DBSCAN algorithm
    from sklearn.cluster import DBSCAN
    
    while True:
        
        print("\n\n#########################################################################")
        print("DBSCAN ALGORITHM")
        print("#########################################################################")
              
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
    start_time = time.time() 
    db = DBSCAN(eps= epsilon, min_samples = minSamples,algorithm = algorithm).fit(X)
    print("\n\nRun Time ->","--- %s seconds ---" % (time.time() - start_time))
    print("Data Successfully Clustered")
    
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    
    Z = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(Z))
    n_noise_ = list(Z).count(-1)
    
    n = -1  # DBSCAN return index -1 cluster
    clusters = []
    while n + 1 < n_clusters:
        clusters.append(n)
        n += 1
    
    #DBSCAN Results
    dbscanR = pd.crosstab(Y,Z)
    maxVal = dbscanR.idxmax()
    
    return Z,clusters,n_noise_,dbscanR,maxVal




def dbF1(Z,Y,clusters,maxVal):#F1 score for DBSCAN
    from sklearn.metrics import f1_score
    #Encoding data to F-score
    
    
    n = 0 # counter
    c = -1 # - counter max Value has negative index
    dictionaryCluster  = {} # creating an empty dictionary 
    f1 = 0
    average = ''
    
    while n < len(clusters):# while counter < number of clusters
        dictionaryCluster[clusters[n]] = maxVal[c] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        c+=1
    
        
    Z[:] = [dictionaryCluster[item] for item in Z[:]] # match key with the index of klabels and replace it with key value
    
    Y = np.array(Y,dtype = int) #Making sure that labels are in a int array
    while True:
        
        average = input("Average Method[weighted,micro,macro]:")
        
        if average == "weighted" or average == "micro" or average == "macro":
            break
        
        else:
            
            print("Error\n\n")
    
    f1 = f1_score(Y,Z, average = average)
    return f1,dictionaryCluster


def dbNMI(Z,Y,clusters,maxVal):
    from sklearn.metrics import normalized_mutual_info_score
    
    n = 0 # counter
    c = -1 # - counter max Value has negative index
    NMI = 0
    dictionaryCluster  = {} # creating an empty dictionary 
    average = ''
    
    while n < len(clusters):# while counter < number of clusters
        dictionaryCluster[clusters[n]] = maxVal[c] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        c+=1
    
    Y = np.array(Y,dtype = int) #Making sure that labels are in a int array

    while True:
        
        average = input("Average Method[geometric,min,arithmetic,max]:")
        
        if average == "geometric" or average == "min" or average == "arithmetic" or average == "max":
            break
        else:
            
            print("Error\n\n")
    
    NMI = normalized_mutual_info_score(Y, Z, average_method= average)
    
    return NMI,dictionaryCluster

def dbARS(Z,Y,clusters,maxVal):
    from sklearn.metrics import adjusted_rand_score
    
    n = 0 # counter
    c = -1 # - counter max Value has negative index
    ars = 0
    dictionaryCluster  = {} # creating an empty dictionary 
    
    while n < len(clusters):# while counter < number of clusters
        dictionaryCluster[clusters[n]] = maxVal[c] #creating key(cluster index) with value (max number of the clustering results) for every iteration
        n+=1
        c+=1
    
    ars = adjusted_rand_score(Y,Z)
    
    return ars,dictionaryCluster


def isolationForest(X,Y):
    from sklearn.ensemble import IsolationForest
    
    while True:
        contamination = input("Contamination[Float 0 to 0.5]: ")
        
        try:
            contamination = float(contamination)
            
        except ValueError:
            
            print("Enter a Number")
            
        if type(contamination) == float and (contamination >= 0 and contamination <= 0.5):
            break
    
    print("\nClustering...\n")   
    
    start_time = time.time() 
    Z = IsolationForest(max_samples = "auto",behaviour = "new",contamination = contamination).fit_predict(X)
    print("\n\nRun Time ->","--- %s seconds ---" % (time.time() - start_time))
    
    Z = np.array(Z,dtype = object)
    
    ifR = pd.crosstab(Y,Z)
    ifR = pd.DataFrame(ifR)
    maxVal = ifR.idxmax()
    
    n = -1  # Isolation Forest return index -1 and 1 cluster
    clusters = []
    while n < len(ifR.columns):
        clusters.append(n)
        n += 2
        
    return Z,ifR,maxVal,clusters

def ifF1(Z,Y,clusters,maxVal):
    from sklearn.metrics import f1_score
    
    n = 0 # counter
    c = -1 # - counter max Value has negative index
    f1 = 0
    average = ''
    dictionaryCluster  = {} # creating an empty dictionary 

    
    while n < len(clusters): # Since we got -1 and 1 clusters , in order to assing the corrects result counter starts at -1 and it increments by 2 so it can have the 1 index of maxLOFvalue
        dictionaryCluster[clusters[n]] = maxVal[c] 
        n+=1
        c+=2
        
    Z[:] = [dictionaryCluster[item] for item in Z[:]] # match key with the index of klabels and replace it with key value
    
    Y = np.array(Y,dtype = int)
    Z = np.array(Z,dtype = int)
    
    while True:
        
        average = input("Average Method[weighted,micro,macro]:")
        
        if average == "weighted" or average == "micro" or average == "macro":
            break
        
        else:
            
            print("Error\n\n")
    
    f1 = f1_score(Y,Z, average = average) #[None, 'micro', 'macro', 'weighted']
    
    return f1,dictionaryCluster
    

def LOF(X,Y):
    from sklearn.neighbors import LocalOutlierFactor 
    
    while True:
        contamination = input("Contamination[Float 0 to 0.5]: ")
        
        try:
            contamination = float(contamination)
            
        except ValueError:
            
            print("Enter a Number")
            
        if type(contamination) == float and (contamination > 0 and contamination <= 0.5):
            break
        
    while True:
        algorithm = input("Algorithm['auto’, ‘ball_tree’, ‘kd_tree’, 'brute']:")
            
        if algorithm == "auto" or algorithm == "ball_tree" or algorithm == "kd_tree" or algorithm == "brute":
            break
        else:
            
            print("Error\n\n")
            
    print("\nClustering...\n")
    
    start_time = time.time() 
    lof = LocalOutlierFactor(contamination = contamination,algorithm = algorithm).fit_predict(X)
    print("\n\nRun Time ->","--- %s seconds ---" % (time.time() - start_time))
    
    lofR = pd.crosstab(Y,lof)
    maxVal = lofR.idxmax()
    
    
    n = -1  # LOF return index -1 and 1 cluster
    clusters = []
    while n < len(lofR.columns):
        clusters.append(n)
        n += 2
    
    
    
    return lof,lofR,maxVal,clusters
    

def lofF1(Z,Y,clusters,maxVal):
    from sklearn.metrics import f1_score
    
    n = 0 # counter
    c = -1 # - counter max Value has negative index
    f1 = 0
    dictionaryCluster  = {} # creating an empty dictionary 
    
    while n < len(clusters): # Since we got -1 and 1 clusters , in order to assing the corrects result counter starts at -1 and it increments by 2 so it can have the 1 index of maxLOFvalue
        dictionaryCluster[clusters[n]] = maxVal[c] 
        n+=1
        c+=2
        
    Z[:] = [dictionaryCluster[item] for item in Z[:]] # match key with the index of klabels and replace it with key value
    Y = np.array(Y,dtype = int)
    Z = np.array(Z,dtype = int)
    while True:
        
        average = input("Average Method[weighted,None,micro,macro]:")
        
        if average == "weighted" or average == "micro" or average == "macro" or average == "None":
            break
        
        else:
            
            print("Error\n\n")
    f1 = f1_score(Y,Z, average = average) #[None, 'micro', 'macro', 'weighted']
    
    return f1,dictionaryCluster

clear()
##########################################################################
path,dataSetOption = getDataSet()
#########################################################################
#########################################################################
dataSet = readingData(path)
#########################################################################
#########################################################################
dataSet = checkMissing(dataSet)
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
data = riskEncodingData(data,dataOption)
#########################################################################
#########################################################################
data = oneHotEncodingData(data,dataOption) #One hot Encode with the complete data
#########################################################################
#########################################################################
data = scaling(data)
#########################################################################
#########################################################################
data = shuffleData(data)
#########################################################################

while True:  
    while True:
        print("\n\n#########################################################################")
        print("Algorithm Menu")
        print("#########################################################################")
        
        print("1.Kmeans")
        print("2.Dbscan")
        print("3.Isolation Forest")
        print("4.Local Factor Outlier")
        
        algorithmOption = input("option:")
        
        if algorithmOption == "1" or algorithmOption == "2" or algorithmOption == "3" or algorithmOption == "4":
                break
        else:
            
            print("Error\n\n")

    
    if algorithmOption == "1":
        #########################################################################
        #KMEANS
        klabels,kClusters,kmeansR,maxKvalue,inertia = kmeansClustering(data,labels)
        print("#########################################################################")
        print("KMEANS RESULTS\n\n")
        print("Clusters -> ",kClusters,"\n")
        print("Inertia -> ",inertia)
        print(kmeansR,"\n\n")
        print("Max True Label","\n\n",maxKvalue)
        print("#########################################################################")
        #########################################################################
        print("\n\n#########################################################################")
        print("Kmeans Score Metrics Menu")
        print("#########################################################################")
        
        while True:
            print("1.F1 Score")
            print("2.Normalized Mutual Info Score")
            print("3.Adjusted Rand Score")
        
            kScoreOption = input("option:")
            
            if kScoreOption == "1" or kScoreOption == "2" or kScoreOption == "3":
                break
            else:
                
                print("Error\n\n")
     
        if kScoreOption == "1":
            #########################################################################
            #F1 Score
            kmeansF1,clusterAssigned = kF1(klabels,labels,maxKvalue,kClusters)
            print("\n\n#########################################################################")
            print("Cluster Matchings by Maximun Intersection[Found: True] -> ",clusterAssigned)
            print("KMEANS F1 Score -> ",kmeansF1)
            print("#########################################################################")
            #########################################################################
        
        elif kScoreOption == "2":
            #########################################################################
            kmeansNMI,clusterAssigned = kNMI(klabels,labels,maxKvalue,kClusters)
            print("\n\n#########################################################################")
            print("Cluster Matchings by Maximun Intersection[Found: True] -> ",clusterAssigned)
            print("KMEANS Normalized Mutual Info Score -> ",kmeansNMI)
            print("#########################################################################")
            #########################################################################
    
        elif kScoreOption == "3":
            
            #########################################################################
            kmeansARS,clusterAssigned = kARS(klabels,labels,maxKvalue,kClusters)
            print("\n\n#########################################################################")
            print("Cluster Matchings by Maximun Intersection[Found: True] -> ",clusterAssigned)
            print("KMEANS Adjusted Rand Score -> ",kmeansARS)
            print("#########################################################################")
            #########################################################################
            
    elif algorithmOption == "2":
        #########################################################################
        #DBSCAN
        dblabels,dbClusters,nNoises,dbscanR,maxDBvalue = dbscanClustering(data,labels) 
        print("#########################################################################")
        print("DBSCAN RESULTS\n\n")
        print("Clusters -> ",dbClusters,"\n")
        print(dbscanR,"\n\n")
        print("Noise -> ",nNoises)
        print("Max True Label","\n\n",maxDBvalue)
        print("#########################################################################")
        #########################################################################
        print("\n\n#########################################################################")
        print("Dscan Score Metrics Menu")
        print("#########################################################################")
        print("1.F1 Score")
        print("2.Normalized Mutual Info Score")
        print("3.Adjusted Rand Score")
        
        while True:
            
            dbScoreOption = input("option:")
            
            if dbScoreOption == "1" or dbScoreOption == "2" or dbScoreOption == "3":
                break
            else:
                
                print("Error\n\n")
    
        if dbScoreOption == "1":
            #########################################################################
            #F1 Score dbscan
            dbscanF1,clusterAssigned = dbF1(dblabels,labels,dbClusters,maxDBvalue)
            print("\n\n#########################################################################")
            print("Cluster Matchings by Maximun Intersection[Found: True] -> ",clusterAssigned)
            print("DBSCAN F1 Score -> ",dbscanF1)
            print("#########################################################################")
            #########################################################################
            
        elif dbScoreOption == "2":
            #########################################################################
            dbscanNMI,clusterAssigned = dbNMI(dblabels,labels,dbClusters,maxDBvalue)
            print("\n\n#########################################################################")
            print("Cluster Matchings by Maximun Intersection[Found: True] -> ",clusterAssigned)
            print("DBSCAN Normalized Mutual Info Score -> ",dbscanNMI)
            print("#########################################################################")
            #########################################################################
            
        elif dbScoreOption == "3":
            #########################################################################
            dbscanARS,clusterAssigned = dbARS(dblabels,labels,dbClusters,maxDBvalue)
            print("\n\n#########################################################################")
            print("Cluster Matchings by Maximun Intersection[Found: True] -> ",clusterAssigned)
            print("DBSCAN Adjusted Rand Score -> ",dbscanARS)
            print("#########################################################################")
            #########################################################################
        
        
    elif algorithmOption == "3":
        #########################################################################
        ifLabels,ifR,MaxIfVal,ifNclusters = isolationForest(data,labels)
        print("#########################################################################")
        print("Isolation Forest RESULTS\n\n")
        print("Clusters -> ",ifNclusters,"\n")
        print(ifR,"\n\n")
        print("Max True Label","\n\n",MaxIfVal)
        print("#########################################################################")
        #########################################################################
        print("\n\n#########################################################################")
        print("Isolation Forest Score Metrics Menu")
        print("#########################################################################")
        print("1.F1 Score")
        
        while True:
            
            ifScoreOption = input("option:")
            
            if ifScoreOption == "1":
                break
            else:
                
                print("Error\n\n")
        
        if ifScoreOption == "1":
            
            ##########################################################################
            isolationForestF1,clusterAssigned = ifF1(ifLabels,labels,ifNclusters,MaxIfVal)
            print("\n\n#########################################################################")
            print("Cluster Matchings by Maximun Intersection[Found: True] -> ",clusterAssigned)
            print("Isolation Forest F1 Score -> ",isolationForestF1)
            print("#########################################################################")
            ##########################################################################
        
    elif algorithmOption == "4":
        #########################################################################
        LOFlabels,lofR,maxLOFvalue,lofClusters = LOF(data,labels)
        print("#########################################################################")
        print("Local Outlier Factor RESULTS\n\n")
        print("Clusters -> ",lofClusters,"\n")
        print(lofR,"\n\n")
        print("Max True Label","\n\n",maxLOFvalue)
        print("#########################################################################")
        #########################################################################
        print("\n\n#########################################################################")
        print("LOF Score Metrics Menu")
        print("#########################################################################")
        print("1.F1 Score")
        
        while True:
            
            lofScoreOption = input("option:")
            
            if lofScoreOption == "1":
                break
            else:
                
                print("Error\n\n")
        
        if lofScoreOption == "1":
            
            ##########################################################################
            LOFf1,clusterAssigned = lofF1(LOFlabels,labels,lofClusters,maxLOFvalue)
            print("\n\n#########################################################################")
            print("Cluster Matchings by Maximun Intersection[Found: True] -> ",clusterAssigned)
            print("LOF F1 Score -> ",LOFf1)
            print("#########################################################################")
            ##########################################################################
                
    while True: # If the user want to Make a new clustering algorithm test
        
        decision = input("Try another Clustering Algorithm[y/n]:")
        
        if decision == "y" or  decision == "n":
            break
        else:
            
            print("Error\n\n")
    
    
    if decision == "n":
        break
    
    else:
        clear()