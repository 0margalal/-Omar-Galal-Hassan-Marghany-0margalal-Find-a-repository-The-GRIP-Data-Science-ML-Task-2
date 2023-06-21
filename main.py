# Omar Galal Hassan Marghany

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# The main objective of this Python code is to build an Unsupervised ML model specifically
# K-means model to predict the optimum number of clusters and represent it visually.
# I Imported the libraries I may need like Sklearn, Pandas, and Numpy.

Data = pd.read_csv(r"C:\Users\Omar Galal Hassan\Desktop\The Grip 2\Iris.csv")
# Now I opened the given CSV file that has two variables ( Id , SepalLengthCm , SepalWidthCm ,
# PetalLengthCm , PetalWidthCm , Species )

print('\n', "\nTop 5 Rows :\n", Data.head(), '\n', '\n')
print("\nDescription about the Data : \n", Data.describe(), '\n')
# First let me show you the data and its description
print('\n', "\nData's Info :\n", Data.info(), '\n')

# Third unsupervised model


Xc = Data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(Xc)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42, max_iter=300)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.style.use('dark_background')
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# WCSS is the sum of squares is a measure of the quality of clustering in KMeans clustering algorithm.
# It represents the sum of the squared distances between each data point and its assigned cluster centroid.

kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0, max_iter=300)

XX = kmeans.fit_predict(Xc)

# selecting the number of clusters where the rate of decrease in WCSS starts to level off
# So we choose 3 clusters because the WCSS starts to decrease
# in the other hand , choosing too much clusters will end up with clusters that
# are too specific and not representative of the overall structure of the data

plt.style.use('dark_background')
plt.scatter(Xc[XX == 0]['SepalLengthCm'], Xc[XX == 0]['SepalWidthCm'], s=100, c='green', label='Iris-setosa')
plt.scatter(Xc[XX == 1]['SepalLengthCm'], Xc[XX == 1]['SepalWidthCm'], s=100, c='red', label='Iris-versicolour')
plt.scatter(Xc[XX == 2]['SepalLengthCm'], Xc[XX == 2]['SepalWidthCm'], s=100, c='blue', label='Iris-virginica')

# Plotting the centroids of the clusters
# In this code snippet, I assumed that you want to scatter plot
# the SepalLengthCm on the x-axis and SepalWidthCm on the y-axis.

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label='Centroids')

plt.legend()
plt.show()
