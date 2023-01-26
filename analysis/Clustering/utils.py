import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as SS   # z-score standardization
from sklearn.cluster import KMeans, DBSCAN               # algorithms
from sklearn.decomposition import PCA                    # dimensionality reduction
from sklearn.metrics import silhouette_score             # to eval cohesion/separation in a cluster
from sklearn.neighbors import NearestNeighbors           # to get optimal epsilon (neighborhood distance) value
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------------------------
# Analysis_Clustering
# ---------------------------------------------------------------------------------------------
# *Note:* all functions are from https://programminghistorian.org/en/lessons/clustering-with-scikit-learn-in-python

# To find a suitable number (k) of clusters based on a given dataset
# when performing k-means clustering
def elbowPlot(range_, data, figsize=(10,10)):
    inertia_list = []
    for n in range_:
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(data)
        inertia_list.append(kmeans.inertia_)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sns.lineplot(y=inertia_list, x=range_, ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Inertia")
    ax.set_xticks(list(range_))
    fig.show()
#     fig.savefig("elbow_plot.png")

# To find the best epsilon (neighborhood distance) based on a given dataset
# when using DBSCAN
def findOptimalEps(n_neighbors, data):
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)

# To iteratively select features based on silhouette score & k-means clustering
def progressiveFeaturesSelection(df, n_clusters=3, max_features=4):
    feature_list = list(df.columns)
    selected_features = list()
    initial_feature = ""
    high_score = 0
    for feature in feature_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=22)
        data_ = df[feature]
        labels = kmeans.fit_predict(data_.to_frame())
        score_ = silhouette_score(data_.to_frame(), labels)
        print("Proposed new feature {} with score {}".format(feature, score_))
        if score_ >= high_score:
            initial_feature = feature
            high_score = score_
    print("Initial feature: {} | Silhouette score: {}".format(initial_feature, high_score))
    feature_list.remove(initial_feature)
    selected_features.append(initial_feature)
    for _ in range(max_features-1):
        high_score = 0
        selected_feature = ""
        print("Starting selection {}...".format(_))
        for feature in feature_list:
            selection_ = selected_features.copy()
            selection_.append(feature)
            kmeans = KMeans(n_clusters=n_clusters, random_state=22)
            data_ = df[selection_]
            labels = kmeans.fit_predict(data_)
            score_ = silhouette_score(data_, labels)
            print("Proposed new feature {} with score {}".format(feature, score_))
            if score_ > high_score:
                selected_feature = feature
                high_score = score_
        selected_features.append(selected_feature)
        feature_list.remove(selected_feature)
        print("Selected new feature {} with score {}".format(selected_feature, high_score))
    return selected_features
        
        
    
    
    
    
    
    