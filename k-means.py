# import the necessary libraries

import math

import numpy as np
import pandas as pd

# import the dataset

# Dataset downloaded from: https://download.data.world/datapackage/environmentdata/iris-species
df = pd.read_csv("iris.csv")

# fetch all features into different sets

id = df['id'].values
f1 = df['sepallengthcm'].values
f2 = df['sepalwidthcm'].values
f3 = df['petallengthcm'].values
f4 = df['petalwidthcm'].values
species = df['species'].values

# Storing original data into an array
original_data = np.array(list(zip(id, species, f1, f2, f3, f4)))

# data array with only the features which will be used to cluster the data
data = np.array(list(zip(f1, f2, f3, f4)))


# ================================================================================================
# ==== Helper functions ==========================================================================
# ================================================================================================


def init_centroids(k, data):
    '''
    This function will be used to initialize the centroids once in the beginning.

    Centroids will be randomly chosen as points(features) from the dataset, as this provides faster
    convergence. Another implementation
    can be assigning completely random numbers as centroids, but this is dangerous.

    :param k: (int) number of centroids
    :param data: (np-array) containing the features of the dataset
    :return: (list) 'k' number of randomly selected centroids from the dataset
    '''
    c = []
    s = np.random.randint(low=1, high=len(data), size=k)
    while (len(s) != len(set(s))):
        s = np.random.randint(low=1, high=len(data), size=k)
    for i in s:
        c.append(data[i])
    return c

def euc_dist(a, b):
    '''
    This function calculates and returns the euclidean distance between two input vectors.

    This is a helper function for cal_dist() to calculate distance of any given point in
    data w.r.t. the centroids.

    :param a: (list) vector a
    :param b: (list) vector b
    :return: (float) euclidean distance between two input vectors
    '''
    sum = 0
    for i, j in zip(a, b):
        a = (i - j) * (i - j)
        sum = sum + a
    return math.sqrt(sum)

def cal_dist(centroids, data):
    '''
    This function will be used to generate the distance table.

    For each point in the given dataset, this function will calculate its euclidean distance
    with respect to each centroid, and a distance table will be generated, which will then be
    used to update the centroid positions.

    :param centroids: (list) containing position of centroids
    :param data: (np-array) containing the features of the dataset
    :return: (list) containing the distances of each point w.r.t. each centroid
    '''
    c_dist = []
    for i in centroids:
        temp = []
        for j in data:
            temp.append(euc_dist(i, j))
        c_dist.append(temp)
    return c_dist

def perf_clustering(k, dist_table):
    '''
    This function will perform clustering on the basis of distance table w.r.t. to all centroids.

    With reference to the distance table, for each point in the table, this function will compare
    its distance from all the centroids, and then cluster the point with the nearest centroid.

    :param k: (int) number of centroids
    :param dist_table: (list) containing the distances of each point w.r.t. each centroid
    :return: (list) containing clusters and indexes of respective members
    '''
    # create empty cluster list of size k
    clusters = []
    for i in range(k):
        clusters.append([])
    # start clustering data points, such that each point is clustered to nearest centroid
    for i in range(len(dist_table[0])):
        d = []
        for j in range(len(dist_table)):
            d.append(dist_table[j][i])
        clusters[d.index(min(d))].append(i)
    return clusters

def update_centroids(centroids, cluster_table, data):
    '''
    This function will update the centroids locations after each iteration

    After performing the clustering, some elements might have migrated from one cluster
    to the another cluster, so w.r.t. the new cluster table, this function will calculate and
    update the new centroid locations

    :param centroids: (list) containing position of centroids
    :param cluster_table: (list) containing clusters and indexes of respective members
    :param data: (np-array) containing the features of the dataset
    :return: (list) containing updated position of centroids
    '''
    for i in range(len(centroids)):
        if (len(cluster_table[i]) > 0):
            temp = []
            for j in cluster_table[i]:
                temp.append(list(data[j]))

            sum = [0] * len(centroids[i])
            for l in temp:
                sum = [(a + b) for a, b in zip(sum, l)]

            centroids[i] = [p / len(temp) for p in sum]
    return centroids


# ================================================================================================
# ==== K-Means ===================================================================================
# ================================================================================================

def kMeans(k, data, max_iterations):
    '''
    Simple implementation of K-Means algorithm

    This function is a very naive implementation of the K-Means algorithm.
    Steps:
    --[1] Initialize random centroids
    --[2] Calculate distances for each point w.r.t. centroid, and store them in distance table.
    --[3] Perform clustering with the available distance table
    --[4] Update the centroid positions
    --[5] Stop if stopping criteria is met, else repeat again from --[2]

    :param k: (int) number of centroids
    :param data: (np-array) containing the features of the dataset
    :param max_iterations: (int) number of maximum iterations allowed.
    :return: none
    '''
    centroids = init_centroids(k, data)
    distance_table = cal_dist(centroids, data)
    cluster_table = perf_clustering(k, distance_table)
    newCentroids = update_centroids(centroids, cluster_table, data)

    for i in range(max_iterations):
        distance_table = cal_dist(newCentroids, data)
        cluster_table = perf_clustering(k, distance_table)
        newCentroids = update_centroids(newCentroids, cluster_table, data)

    # Display the final results
    for i in range(len(newCentroids)):
        print("Centroid #", i, ": ", newCentroids[i])
        print("Members of the cluster: ")
        for j in range(len(cluster_table[i])):
            print(original_data[cluster_table[i][j]])


# Run the K-Means algorithm on the Iris-Dataset with k = 3, and max-iterations limited to 100

kMeans(3, data, 100)
