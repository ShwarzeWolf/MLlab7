import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

colors = ['r', 'g', 'b']
numberOfClusters = 3

def draw_data(arr, massive, message):
    result = PCA(n_components=2).fit_transform(arr)

    x = []
    y = []
    for i in result:
        x.append(i[0])
        y.append(i[1])

    for i in range(numberOfClusters):
        x_i = np.array([x[j] for j in range(len(x)) if massive[j] == i])
        y_i = np.array([y[j] for j in range(len(y)) if massive[j] == i])
        plt.scatter(x_i, y_i, s=7, c=colors[i])

    plt.suptitle(message)
    plt.show()


def kMeansClassifier(data, k):
    current_centers = np.random.rand(k, len(data[0]))

    current_distances = np.zeros(k)
    points_clusters = np.zeros(len(data), dtype=int)

    while True:

        for i in range(len(data)):
            for j in range(k):
                current_distances[j] = np.linalg.norm(data[i] - current_centers[j], axis=0)

            points_clusters[i] = np.argmin(current_distances)

        previous_centers = deepcopy(current_centers)

        for i in range(k):
            points = [data[j] for j in range(len(data)) if points_clusters[j] == i]

            if len(points):
                current_centers[i] = np.mean(points, axis=0)

        if np.array_equal(previous_centers, current_centers):
            return points_clusters, current_centers


data = pd.read_csv('dataset_61_iris.csv')
data['class'] = LabelEncoder().fit_transform(data['class'])

target = list(data['class'])

data = data.drop('class', 1)
data = minmax_scale(data)

draw_data(data, target, "real data")

clusters, centers = kMeansClassifier(data, numberOfClusters)
draw_data(data, clusters, "predicted values")

clusters_matched = 0
for i in range(numberOfClusters):
    cluster_class = np.zeros(numberOfClusters, dtype=int)

    for j in range(len(data)):
        if clusters[j] == i:
            cluster_class[target[j]] += 1

    clusters_matched += np.amax(cluster_class)

purity = clusters_matched / len(data)

print("purity:", end=' ')
print(purity)

cohesions = []
separetions = []

for i in range(9):
    clusters, centers = kMeansClassifier(data, i + 1)

    currentCohesion = 0
    for j in range(len(data)):
        currentCohesion += np.linalg.norm(centers[clusters[j]] - data[j], axis=0)

    sum = 0
    for j in range(len(data)):
        sum += centers[clusters[j]]
        sum /= 150

    currentSeparation = 0

    for k in range(i):
        currentSeparation += np.linalg.norm((centers[k] - sum) * (centers[k] - sum))

    cohesions.append(currentCohesion)
    separetions.append(currentSeparation / (i + 1))

plt.plot([i for i in range(1, 10)], cohesions)
plt.suptitle('Cohesion')
plt.show()

plt.plot([i for i in range(1, 10)], separetions)
plt.suptitle('separatons')
plt.show()


print("cohesions:", end = ' ')
print(cohesions)

print("separations:", end = ' ')
print(separetions)