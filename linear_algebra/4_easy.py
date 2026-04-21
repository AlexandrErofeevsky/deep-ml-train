def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == "column":
        means = [sum(col) / len(col) for col in zip(*matrix)]
    else:
        means = [sum(row) / len(row) for row in matrix]
    return means
