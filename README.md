# Clustering Based Anomaly Detection

## Description
This clustering based anomaly detection project implements unsupervised clustering algorithms on the [NSL-KDD](https://pdfs.semanticscholar.org/1b34/80021c4ab0f632efa99e01a9b073903c5554.pdf) and [IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html) datasets. The project includes options for preprocessing the datasets. It then clusters the datasets, mainly using the K-means and DBSCAN algorithms. Finally, it evaluates the clustering performed by the algorithms using standard metrics such as F-Score.

## Requirements

* [Python >= 3.5](https://www.python.org/)
* [Anaconda](https://www.anaconda.com/distribution/)
* [Scikit](https://scikit-learn.org/stable/install.html)
* [SciPy](https://www.scipy.org/#)
* [NumPy](http://numpy.org/)
* [joblib](https://joblib.readthedocs.io/en/latest/#)
* [pandas](https://pandas.pydata.org/)
* [Spyder environment](https://www.spyder-ide.org/)

## Installation

For this project, we installed Anaconda-Navigator to use as our package and environment manager. Under the Environments tab in Anaconda, we created an environment and downloaded the libraries listed in the prerequisites for this project.
This [guide](https://docs.anaconda.com/_downloads/9ee215ff15fde24bf01791d719084950/Anaconda-Starter-Guide.pdf) can help use Anaconda


## Code Details
After you install all the requirements you should be able to run the code without any problems. This code is implemented to be user friendly and the steps will be briefly explained below:

##### 1. Dataset option
* ![image](https://user-images.githubusercontent.com/31083873/62171123-263b7400-b2eb-11e9-92ea-27dd3511b052.png)
The user is asked to input which dataset will be analyzed in this run of the anomaly detetion algorithms. The two datasets that this project used contain different types of data and therefore require different types of preprocessing; thus, the user must choose which dataset to preprocess before beginning anomaly detection.

##### 2. Path
* ![image](https://user-images.githubusercontent.com/31083873/62171230-816d6680-b2eb-11e9-814b-d6d2d2f819dd.png)
The user is asked to input the path of the data set. After [downloading the dataset](https://www.unb.ca/cic/datasets/index.html) to your computer, copy the path to that dataset and input the path here.

##### 3. Variable Menu
* ![image](https://user-images.githubusercontent.com/31083873/62171295-afeb4180-b2eb-11e9-8958-317cc71b9e43.png)
The user is asked to choose the variables he wants to be working on.
As explained in step 1, the two data sets have different types of features. Specifically, the NSL-KDD Dataset has categorical data that must either be converted into numerical data or eliminated. The user can choose between three options for dealing with the categorical features on the NSL-KDD Dataset:

  1. The data will have categorical features(protocols,service type,attack types,service error) and the data within those features will be [one hot  encoded](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)(encode categorical features into numerical features)

  2. The categorical features are removed from the data.

  3. The Categorical features (service type, attack types, and service error/flag) are encoded with [Risk Values](http://www.jatit.org/volumes/Vol65No3/13Vol65No3.pdf). Since protocols do not have associated risk values, they are one hot encoded

##### 4. Encoding Menu
* ![image](https://user-images.githubusercontent.com/31083873/62171931-ed50ce80-b2ed-11e9-9963-45de4cc4301e.png)
The user is asked to encode the labels. The NSL-KDD Dataset contains 22 usual attack types plus the normal category for a total of 23 possible labels.
  1. The labels are converted in binary labels (normal and abnormal). Every attack name that is not normal - in other words, that is an attack - is renamed with the label abnormal. After that, the labels are encoded into binary numbers where 0 is normal and 1 is abnormal. Because we can't calculate a metric score with categorical features, so the normal and abnormal labels must be converted to numeric data.

  2. The labels are converted into a 5 main categoires (normal,DoS,Probe,U2R,R2L) using the information provided in [this analysis of the dataset](https://pdfs.semanticscholar.org/1b34/80021c4ab0f632efa99e01a9b073903c5554.pdf). After that, each attack is encoded into one of 5 numbers where normal is 0, Dos is 1, Probe is 2, R2L is 3 and U2R is 4.

##### 5. Scale the data
* ![image](https://user-images.githubusercontent.com/31083873/62172317-1756c080-b2ef-11e9-873b-3c4a0f8fb0e9.png)
The user is asked if he or she wants to Scale the data. We use [Min Max Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html). We do this because we want our data instances to be at the same range, and Min Max Scaler puts the data in a range of [0,1] by feature. This allows the distance-based anomaly detection algorithms to accurately analyze the data.

##### 6. Shuffle the data
* ![image](https://user-images.githubusercontent.com/31083873/62183286-db375600-b316-11e9-97e4-71f1440ee1ed.png)
The user is asked if he or she wants to suffle the data. Because one of the clustering algorithms, namely DBSCAN, could potentially return a different clustering depending on the order of the dataset, we attempted to shuffle the data and compare results. Unfortunately, the shuffled data returned clusters vastly different from the unshuffled data, with enough reason to believe that the shuffling algorithm was not working properly. Users are welcome to attempt shuffling the data but are forewarned that the shuffling may not return desired results.

##### 7. Algorithm Menu
* ![image](https://user-images.githubusercontent.com/31083873/62183597-0ff7dd00-b318-11e9-9bcf-d26b4f6ae0ac.png)
The user is asked which anomaly detection algorithm he or she wants to use on the data. Each algorithm is discussed in greater detail in the Analyzing Dataset section.

Each algorithm requires user-input parameters.

  ###### K-Means
     ###### Initialization method
* ![image](https://user-images.githubusercontent.com/31083873/62186624-2b68e500-b324-11e9-9fdb-c700ee87ee4c.png)
K-Means provides different options for choosing the initial cluster centers. In this project, the user can choose either the random method or SciKitLearn's more sophisticated [k-means++](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) method.
      ###### Clusters
* ![image](https://user-images.githubusercontent.com/31083873/62186784-97e3e400-b324-11e9-8505-d35d78ee9fc1.png)
Users must choose the number of clusters for K-Means. The elbow method is popular for choosing the number of clusters. Read  more below in the Analyzing Dataset section.
  
  ###### DBSCAN  
  * ![image](https://user-images.githubusercontent.com/31083873/62664174-cfabe680-b937-11e9-8352-d9cd5550c7f3.png)
  DBSCAN need 2 main parameter epsilon and min samples , the algorithm parameter will affect the run time, we concluded that    brute is the fastest one for the NSL-KDD dataset.
  
  ###### Local Outlier Factor
  * ![image](https://user-images.githubusercontent.com/31083873/62664862-65487580-b93a-11e9-80e5-32dcff8b0ac1.png)
  Users must choose the ratio of anomalies in the dataset. This is called the contamination factor.
 
  ###### Isolation Forest 
  * ![image](https://user-images.githubusercontent.com/51713553/62648301-c149d480-b90f-11e9-848f-1fbe843099cb.png)
  Users must choose the ratio of anomalies in the dataset. This is called the contamination factor.
  
##### 8. Scoring Metrics
* ![image](https://user-images.githubusercontent.com/31083873/62186832-be098400-b324-11e9-9036-ae5413a4535e.png)
  
* ![image](https://user-images.githubusercontent.com/51713553/62640889-bdae5180-b8ff-11e9-975d-f2c356561180.png)
Kmeans F1-score
  
  
* ![image](https://user-images.githubusercontent.com/31083873/62664455-cb33fd80-b938-11e9-8032-72bb83af578d.png)
DBSCAN F1-score
 


### Preprocessing Dataset

This project was designed to be used with the NSL-KDD and IDS 2017 datasets, available for download [here](https://www.unb.ca/cic/datasets/index.html). The preprocessing options thus are specific for each dataset. 

The NSL-KDD dataset has [categorical data](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) that must be omitted or encoded as numerical data to be clustered. The options in this project for dealing with categorical data include omitting categorical features, One Hot Encoding catgorical features, and [assigning risk values](http://www.jatit.org/volumes/Vol65No3/13Vol65No3.pdf) to Server Type and Flag features while One Hot Encoding Protocol data. One Hot Encoding is a process that takes each categorical option in a feature and makes that option a feature itself, assinging each data instance a "0" if it does not contain that category and a "1" if it does. While this option allows the user to keep the structure of the categorical data without assigning arbitrary hierarchical ordering to categories, this option also increases the number of features and thus is not always optimal for already-large datasets. For this reason, the code offers three different methods of dealing with categorical data.

The IDS-2017 dataset has missing values that must be dealt with as well. The code offers the user the option of deleting the columns with missing values, imputing "0", imputing the mean, median, or mode of the feature, or using the Iterative Imputer method offered by Python.

The interactive code asks the user to specify which of the two datasets he or she is using.

### Analyzing Dataset

The code offers four different anomaly detection algorithms, namely K-Means, DBSCAN, Local Outlier Factor (LOF), and Isolation Forest. K-Means and DBSCAN are clustering algorithms, while LOF is a K-Nearest-Neighbor algorithm and Isolation Forest is a decision tree algorithm, both using a contamination factor to classify data as normal or anomaly.

[K-Means](https://www.youtube.com/watch?v=_aWzGGNrcic) clusters data by starting with user-specified K initial cluster centroids, and assigning all points to the nearest centroid. Based on the assignments, the algorithm recalculates the cluster centers and reassigns all points to the nearest cluster center. The algorithm repeats this process for a default of 300 iterations. When the process ends, K-Means has clustered data into K clusters. [SciKitLearn's K-Means algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) offers the option for the user to also specify the method for initialization, the way that the algorithm chooses which points to use as initial cluster centroids. In this project, the user specifies K, the number of initial cluster centroids and eventual clusters. A typical way of choosing K is often by the [elbow method](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html). The implementation of K-Means in this project reports the sum of squared distances to cluster centers (or squared sum of errors, SSE) needed in the elbow plot, so a user can run tests with different values of K and plot that against the SSE for each K value. A user can then subjectively choose the elbow point on such a plot to determine the best K, and can then conduct tests with this K. The researchers suggest using a few values of K around the elbow and comparing the evaluation metric scores generated for each K in order to determine the best value of K.

[Density-Based Spacial Clustering of Applications with Noise](https://medium.com/@elutins/dbscan-what-is-it-when-to-use-it-how-to-use-it-8bd506293818), or DBSCAN, relies on two user-input parameters, namely epsilon and minimum samples. Epsilon denotes the neighborhood of density to be explored for each data point, and minimum samples denote the minimum number of samples needed to be within a point’s epsilon neighborhood for said point to be considered a core point. Points within another core point’s epsilon neighborhood, but not core points themselves, are considered border points. Meanwhile, points that are not within another core point’s epsilon neighborhood, and that are not core points themselves, are considered anomalous points or noise. DBSCAN finds clusters of core points and border points and reports those clusters along with a group of all of the anomalous points. [SciKitLearn's DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) offers the user other parameters to manipulate the specific way that DBSCAN calculates the clusters; this project uses all default parameters except for the algorithm parameter, for which the project specifies the 'brute' option in order to reduce run time.
**DBSCAN run time will depend of how big the dataset is and what resources your computer has. Since "DBSCAN groups together points that are close to each other based on a distance measurement," it is slower than K-means algorithm (Salton do Prado, 2017). The experiments on DBSCAN were made on a Macbook Pro 2.6 GHz i7 with 16 GB of RAM memory and using the Brute parameter for the algorithm. The average time for these experiments was 3 minutes. DBSCAN tests were attempted on a Macbook air 1.6 GHz i5 with 8GB of RAM, but after 30 minutes never finished due to the processing capacity of the computer. Before running experiments with DBSCAN make sure the computer can handle it.**

[Local Outlier Factor](https://towardsdatascience.com/local-outlier-factor-for-anomaly-detection-cc0c770d2ebe), or LOF, begins with the parameter K, a default-set or user-chosen integer. For a specific point, the algorithm calculates the reach-distance to each point, which is essentially the distance from a specific point to another point with a small smoothing caveat for close points. The algorithm then takes the average of the reach-distances for a specific point to each of that point's k-nearest neighbors. The inverse of this average is called the Local Reachability Distance, or LRD. A point's high LRD indicates that the point exists in a highly dense neighborhood and does not have to travel far to encounter all K nearest neighbors, and a point's low LRD indicates the opposite, a low-density neighborhood. The algorithm calculates the LRDs for each point, and finds the average of all LRDs. Finally, the algoirthm calculates the Local Outlier Factor for each point by dividing that point's LRD by the average LRD of all points. An LRD around 1 indicates a point with average density, and an LRD much greater than 1 indicates a point in a much lower-density neighborhood than the average point, and therefore a point that is likely an anomaly. [SciKitLearn's LOF algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) returns the negative of each point's local outlier factor. In this code, one can choose an Offset value such that all points with an LOF more negative than that Offset value are labeled as anomalous points, and all points equal to or more positive than that Offset value are labeled as normal points. 

Similarly to Local Outlier Factor, [Isolation Forest](https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e) returns for each point a score representing the probability of that particular point being an anomaly, and the user must choose a threshold for which scores will indicate an anomaly and which will indicate a normal instance. The algorithm generates the probability scores for each instance by the following process: _First, randomly choose a feature (column). Next, randomly choose a value between the min and max of that feature. Partition, or split the data into two groups: those with values in that feature above the randomly chosen value, and those with values below. Now, choose one of the two groups again and split the data on a random point. Repeat until a single point is isolated. Obtain the number of splits required to isolate that point. Repeat this process, eventually isolating all points across many features, and obtain for each specific point the average number of splits required for that point to be isolated_. The theory behind Isolation Forest states that anomalies occur less frequently and differ more greatly than normal points, and therefore will require fewer partitions, or splits, to isolate them than normal points would require. Thus, a score based on the average number of splits, also known as the average path length, denotes the probability that a particular point is an anomaly. The score is adjusted such that a a score near 1 denotes a likely anomaly, and a score near 0.5 denotes a likely normal point. Again, the user can set the contamination factor to indicate the threshold for scores labeled as anomaly and as normal.


### Evaluating Clusters

The code also offers multiple evaluation metrics for the user to choose from. Each metric depends on comparing the labels of the actual dataset with the labels given by the clustering, or the "true labels" with the "found labels". For both the NSL-KDD and the IDS 2017 datasets, both binary and multiclass labels are available to compare with as "true labels." Users can specify their preference in the interactive code. In this code, users can verify the clustering on their data by using one of three different metrics, namely F-1 Score, Normalized Mutual Information Score (NMI), and Adjusted Rand Score (ARS).  

[F-Score](https://deepai.org/machine-learning-glossary-and-terms/f-score) is the harmonic mean between precision and recall. Precision is the ratio of correctly predicted positive values to all values predicted to be positive. In other words, precision indicates how sure the algorithm is that the found positive values are actually positive. Meanwhile, recall is the ratio of correctly predicted positive values to all values that are actually positive. In other words, recall indicates how sure the algorithm is that it did not miss any positive values in its positive labelings. One can weight either precision or recall to have more influence in the F-Score by changing the beta value in the F-beta function; however, this project opts to keep the weight between precision and recall equal by using the F-1 score. 

The [Normalized Mutual Information Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html) is based on the probability function of entropy and uncertainty between the true and the found labels. 

Instead of measuring entropy as the NMI score does, the [Adjusted Rand Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html) measures the similarity between the true labels and the found labels. Furthermore, the ARS is adjusted for chance, while the NMI is not.


## Roadmap

* Implement Dimensionality Reduction- The main idea to implement this is because both datasets are considerably big  and they consume a lot of processing resources also we want to implement this because we can run DBSCAN algorithm on a bigger data set than IDS2017 and we want to know if we are going to be able to implement the algorithm.

## Poster
![CBAD-Poster](https://user-images.githubusercontent.com/31083873/70267654-41c0fa80-1775-11ea-9fa4-2bc85b1a57a3.png)



## Authors and acknowledgment
* Jeremy Perez
* Bethany Danner
* **Special thanks to Dr. Veronika Neeley for mentoring us throughout this project, and for Dr. Clem Izurieta for organizing the REU program at Montana State University. This work was funded by the [National Science Foundation](https://www.nsf.gov/)**.

## License

MIT License

Copyright (c) 2019 Jeremy Perez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Project status

Current Bugs: After shuflleling the data results are not as excpected
