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

The code offers four different clustering algorithms, namely K-Means, DBSCAN, LOF, and Isolation Forest. **Add links / explanations to these**.

### Evaluating Clusters

The code also offers multiple evaluation metrics for the user to choose from. Each metric depends on comparing the labels of the actual dataset with the labels given by the clustering, or the "true labels" with the "found labels". For both the NSL-KDD and the IDS 2017 datasets, both binary and multiclass labels are available to compare with as "true labels." Users can specify their preference in the interactive code.

The most tested metric used in this code is the [F-Score](https://deepai.org/machine-learning-glossary-and-terms/f-score). **write about other metrics**

## Roadmap


## Contributing

## Authors and acknowledgment

## License


## Project status





 
 
  
  
