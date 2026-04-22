"""
Calculate Covariance Matrix
 - Easy
 - Statistics

Write a Python function to calculate the covariance matrix for a given set of vectors.
The function should take a list of lists,
where each inner list represents a feature with its observations,
and return a covariance matrix as a list of lists.

Example:
    Input:
        [[1, 2, 3], [4, 5, 6]]
    Output:
        [[1.0, 1.0], [1.0, 1.0]]
Reasoning:
    The covariance between the two features is calculated based on their deviations from the mean.
    For the given vectors, both covariances are 1.0, resulting in a symmetric covariance matrix.
"""


def calculate_covariance_matrix(
	vectors: list[list[float]],
) -> list[list[float]]:
	means = [sum(vec) / len(vec) for vec in vectors]
	m = len(vectors[0])

	def cov(idx1: int, idx2: int) -> float:
		return sum(
			(v1 - means[idx1]) * (v2 - means[idx2])
			for v1, v2 in zip(vectors[idx1], vectors[idx2])
		) / (m - 1)

	return [
		[
			cov(row_idx, col_idx) for row_idx in range(len(vectors))
		] for col_idx in range(len(vectors))
	]
