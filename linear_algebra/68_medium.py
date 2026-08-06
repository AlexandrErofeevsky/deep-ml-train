"""
Find the column space of a matrix
 - Medium
 - Linear Algebra

Task: Compute the Column Space of a Matrix
In this task, you are required to implement a function matrix_image(A)
that calculates the column space of a given matrix A.
The column space, also known as the image or span, consists of all linear
combinations of the columns of A. To find this, you'll use concepts from linear
algebra, focusing on identifying independent columns that span the matrix's image.
Your task: Implement the function matrix_image(A) to return the basis vectors
that span the column space of A. These vectors should be extracted from
the original matrix and correspond to the independent columns.

Example:
    Input:
        matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        print(matrix_image(matrix))
    Output:
        [[1, 2],
         [4, 5],
         [7, 8]]
Reasoning:
    The matrix has rank 2, meaning only 2 columns are linearly independent.
    The column space is spanned by the first two column vectors [1, 4, 7] and [2, 5, 8].
    The output matrix contains these two independent columns.
"""
