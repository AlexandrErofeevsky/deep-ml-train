"""
Matrix times Matrix
 - Medium
 - Linear Algebra

multiply two matrices together (return -1 if shapes of matrix don't align),
i.e. C = A ⋅ B

Example:
    Input:
        A = [[1,2],[2,4]], B = [[2,1],[3,4]]
    Output:
        [[ 8,  9],[16, 18]]
Reasoning:
    1*2 + 2*3 = 8; 2*2 + 3*4 = 16; 1*1 + 2*4 = 9; 2*1 + 4*4 = 18

Example 2:
    input: A = [[1,2], [2,4]], B = [[2,1], [3,4], [4,5]]
    output: -1
reasoning:
    the length of the rows of A does not equal the column length of B
"""


def matrixmul(
    a: list[list[int | float]],
    b: list[list[int | float]],
) -> list[list[int | float]] | int:
    if len(a[0]) != len(b):
        return -1

    def vector_dot(v1: list[int | float], v2: list[int | float]) -> int | float:
        return sum(x * y for x, y in zip(v1, v2))

    b_t: list[list[int | float]] = list(zip(*b))
    c = [[vector_dot(row_a, col_b) for col_b in b_t] for row_a in a]
    return c
