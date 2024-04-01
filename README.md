# 95-865: Unstructured Data Analytics

## Table of Contents
1. 
2. 
3. 
4. 
5. 
6. 
7. [Clustering](#7-clustering)






## 7. Clustering

### Overview

- **Clustering** is a method of unsupervised learning, a type of machine learning where the system learns to **identify patterns without prior labeling of the data**.

- Clustering methods aim to group together data points that are "similar" into "clusters", while having different clusters be "dissimilar".
    - Similarity is inversely related to distance (two points being more similar $\rightarrow$ closer in distance)
    - Use **Euclidean distance** between feature vectors

- Clustering structure often occurs
    - 2-D t-SNE plot of handwritten digit images shows clumps that correspond to real digits
    - Crime might happen more often in specific hot spots


### Drug Consumption Data
Source: https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified

#### Demo



### The Art of Defining Similarity/Distance







### K-Means Clustering

1. **Initialization**: Start by selecting $k$ initial centroids randomly.
    - One common approach is to randomly choose $k$ data points from the dataset as the initial centroids.

2. **Assignment**: Assign each data point to the nearest centroid.
    - The most common distance metric used is the Euclidean distance.

3. **Update**: Recalculate the centroids of the clusters by taking the mean of all the data points assigned to each cluster.
    - The mean (center of mass) becomes the new centroid of each cluster.

4. **Iteration**: Repeat the Assignment and Update steps until one of the following conditions is met:
    - The centroids do not change (or below a certain threshold), indicating convergence.
    - The assignments of data points to clusters remain the same between iterations.
    - A predefined number of iterations has been reached.


### K-Means++ 
Aimss to improve the convergence of the K-Means algorithm by carefully choosing the initial centroids.


Probability decays as the distance decreases.



### Gaussian Mixture Model (GMM)





### Caveat
All the models are wrong, rather approximation of the reality.
