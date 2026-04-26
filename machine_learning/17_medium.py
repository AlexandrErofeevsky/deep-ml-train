"""
K-Means Clustering
 - Medium
 - Machine Learning

Your task is to write a Python function that implements the k-Means clustering algorithm.
This function should take specific inputs and produce a list of final centroids.
k-Means clustering is a method used to partition n points into k clusters.
The goal is to group similar points together and represent each group by its center (called the centroid).

Function Inputs:
    points: A list of points, where each point is a tuple of coordinates
    (e.g., (x, y) for 2D points or (x, y, z) for 3D points).
    All points must have the same dimensionality.
    k: An integer representing the number of clusters to form
    initial_centroids: A list of initial centroid points,
    each a tuple of coordinates with the same dimensionality as the input points
    max_iterations: An integer representing the maximum number of iterations to perform
Function Output:
    A list of the final centroids of the clusters,
    where each centroid is rounded to the nearest fourth decimal.

Example:
    Input:
        points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], k = 2, initial_centroids = [(1, 1), (10, 1)], max_iterations = 10
    Output:
        [(1, 2), (10, 2)]
Reasoning:
    Given the initial centroids and a maximum of 10 iterations,
    the points are clustered around these points,
    and the centroids are updated to the mean of the assigned points,
    resulting in the final centroids which approximate the means of the two clusters.
    The exact number of iterations needed may vary,
    but the process will stop after 10 iterations at most.
"""
from collections import defaultdict
from statistics import mean


def k_means_clustering(
    points: list[tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int
) -> list[tuple[float, ...]]:
    def calculate_euclidean(p1, p2):
        return sum((x1 - x2) ** 2 for x1, x2 in zip(p1, p2)) ** (1/2)

    clusters = defaultdict(list)

    final_centroids = initial_centroids.copy()

    for _ in range(max_iterations):
        for point in points:
            distances = [
                calculate_euclidean(point, center)
                for center in initial_centroids
            ]
            min_dist = distances.index(min(distances))
            clusters[min_dist].append(point)

        stop_iterations = True

        for cluster_idx in range(k):
            if clusters[cluster_idx]:
                center = tuple(
                    mean(column) for column in zip(*clusters[cluster_idx])
                )
                if center != final_centroids[cluster_idx]:
                    final_centroids[cluster_idx] = center
                    stop_iterations = False
        if stop_iterations:
            break

    return final_centroids


print(
    k_means_clustering(
        points=[(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)],
        k=2,
        initial_centroids=[(1, 1), (10, 1)],
        max_iterations=10,
    ),
)
