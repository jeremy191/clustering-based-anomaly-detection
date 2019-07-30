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

### Preprocessing Dataset

This project was designed to be used with the NSL-KDD and IDS 2017 datasets, available for download [here](https://www.unb.ca/cic/datasets/index.html). The preprocessing options thus are specific for each dataset. 

The NSL-KDD dataset has [categorical data](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) that must be omitted or encoded as numerical data to be clustered. The options in this project for dealing with categorical data include omitting categorical features, One Hot Encoding catgorical features, and [assigning risk values](http://www.jatit.org/volumes/Vol65No3/13Vol65No3.pdf) to Server Type and Flag features while One Hot Encoding Protocol data.

The IDS-2017 dataset has missing values that must be dealt with as well. The code offers the user the option of deleting the columns with missing values, imputing "0", imputing the mean, median, or mode of the feature, or using the MICE method. (**As of 7/23/2019, MICE Method is not implemented yet.**) 

The interactive code asks the user to specify which of the two datasets he or she is using.

### Clustering Dataset

The code offers four different clustering algorithms, namely K-Means, DBSCAN, LOF, and Isolation Forest.

[K-Means](https://www.youtube.com/watch?v=_aWzGGNrcic) clusters data by starting with user-specified K initial cluster centroids, and assigning all points to the nearest centroid. Based on the assignments, the algorithm recalculates the cluster centers and reassigns all points to the nearest cluster center. The algorithm repeats this process for a default of 300 iterations. When the process ends, K-Means has clustered data into K clusters. [SciKitLearn's K-Means algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) offers the option for the user to also specify the method for initialization, the way that the algorithm chooses which points to use as initial cluster centroids. In this project, the user specifies K, the number of initial cluster centroids and eventual clusters. A typical way of choosing K is often by the [elbow method](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html), **but the researchers of this project instead uses experiments on the F-Scores of clustering with varied K in order to determine the optimal K, so this project does not offer the elbow method.** 

[Density-Based Spacial Clustering of Applications with Noise](https://medium.com/@elutins/dbscan-what-is-it-when-to-use-it-how-to-use-it-8bd506293818), or DBSCAN, relies on two user-input parameters, namely epsilon and minimum samples. Epsilon denotes the neighborhood of density to be explored for each data point, and minimum samples denote the minimum number of samples needed to be within a point’s epsilon neighborhood for said point to be considered a core point. Points within another core point’s epsilon neighborhood, but not core points themselves, are considered border points. Meanwhile, points that are not within another core point’s epsilon neighborhood, and that are not core points themselves, are considered anomalous points or noise. DBSCAN finds clusters of core points and border points and reports those clusters along with a group of all of the anomalous points. [SciKitLearn's DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) offers the user other parameters to manipulate the specific way that DBSCAN calculates the clusters; this project uses all default parameters except for the algorithm parameter, for which the project specifies the 'brute' option in order to reduce run time. 

[Local Outlier Factor](https://towardsdatascience.com/local-outlier-factor-for-anomaly-detection-cc0c770d2ebe), or LOF, begins with the parameter K, a default-set or user-chosen integer. For a specific point, the algorithm calculates the reach-distance to each point, which is essentially the distance from a specific point to another point with a small smoothing caveat for close points. The algorithm then takes the average of the reach-distances for a specific point to each of that point's k-nearest neighbors. The inverse of this average is called the Local Reachability Distance, or LRD. A point's high LRD indicates that the point exists in a highly dense neighborhood and does not have to travel far to encounter all K nearest neighbors, and a point's low LRD indicates the opposite, a low-density neighborhood. The algorithm calculates the LRDs for each point, and finds the average of all LRDs. Finally, the algoirthm calculates the Local Outlier Factor for each point by dividing that point's LRD by the average LRD of all points. An LRD around 1 indicates a point with average density, and an LRD much greater than 1 indicates a point in a much lower-density neighborhood than the average point, and therefore a point that is likely an anomaly. [SciKitLearn's LOF algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) returns the negative of each point's local outlier factor. In this code, one can choose an Offset value such that all points with an LOF more negative than that Offset value are labeled as anomalous points, and all points equal to or more positive than that Offset value are labeled as normal points. 

Similarly to Local Outlier Factor, [Isolation Forest](https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e) returns for each point a score representing the probability of that particular point being an anomaly, and the user must choose a threshold for which scores will indicate an anomaly and which will indicate a normal instance. The algorithm generates the probability scores for each instance by the following process: _First, randomly choose a feature (column). Next, randomly choose a value between the min and max of that feature. Partition, or split the data into two groups: those with values in that feature above the randomly chosen value, and those with values below. Now, choose one of the two groups again and split the data on a random point. Repeat until a single point is isolated. Obtain the number of splits required to isolate that point. Repeat this process, eventually isolating all points across many features, and obtain for each specific point the average number of splits required for that point to be isolated_. The theory behind Isolation Forest states that anomalies occur less frequently and differ more greatly than normal points, and therefore will require fewer partitions, or splits, to isolate them than normal points would require. Thus, a score based on the average number of splits, also known as the average path length, denotes the probability that a particular point is an anomaly. The score is adjusted such that a a score near 1 denotes a likely anomaly, and a score near 0.5 denotes a likely normal point. Again, the user can set the contamination factor to indicate the threshold for scores labeled as anomaly and as normal.


### Evaluating Clusters

The code also offers multiple evaluation metrics for the user to choose from. Each metric depends on comparing the labels of the actual dataset with the labels given by the clustering, or the "true labels" with the "found labels". For both the NSL-KDD and the IDS 2017 datasets, both binary and multiclass labels are available to compare with as "true labels." Users can specify their preference in the interactive code. In this code, users can verify the clustering on their data by using one of three different metrics, namely F-1 Score, Normalized Mutual Information Score (NMI), and Adjusted Rand Score (ARS).  

[F-Score](https://deepai.org/machine-learning-glossary-and-terms/f-score) is the harmonic mean between precision and recall. Precision is the ratio of correctly predicted positive values to all values predicted to be positive. In other words, precision indicates how sure the algorithm is that the found positive values are actually positive. Meanwhile, recall is the ratio of correctly predicted positive values to all values that are actually positive. In other words, recall indicates how sure the algorithm is that it did not miss any positive values in its positive labelings. One can weight either precision or recall to have more influence in the F-Score by changing the beta value in the F-beta function; however, this project opts to keep the weight between precision and recall equal by using the F-1 score. 

The [Normalized Mutual Information Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html) is based on the probability function of entropy and uncertainty between the true and the found labels. 

Instead of measuring entropy as the NMI score does, the [Adjusted Rand Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html) measures the similarity between the true labels and the found labels. Furthermore, the ARS is adjusted for chance, while the NMI is not.

## Roadmap

* Implement Dimensionality Reduction- The main idea to implement this is because both datasets are considerably big  and they consume a lot of processing resources also we want to implement this because we can run DBSCAN algorithm on a bigger data set than IDS2017 and we want to know if we are going to be able to implement the algorithm.

## Contributing

## Authors and acknowledgment
* Jeremy Perez
* Bethany Danner
* **Special thanks to Dr. Veronika Neeley for mentoring us throughout this project, and for Dr. Clem Izurieta for organizing the REU program at Montana State University. This work was funded by the NSF**.

## License

## Project status

Current Bugs:

Rarely, after running an F1-Score, when user tries to run another scoring metric, that metric returns nonsense results - numbers larger than 1 or equal to 0, when a number between 0 and 1 and likely higher than 0.2 is expected. The most occuring nonsense result for NMI is 3.75 and for ARS is 0.0. 
