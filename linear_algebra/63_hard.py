"""
Implement the Conjugate Gradient Method for Solving Linear Systems
 - Hard
 - Linear Algebra

Task: Implement the Conjugate Gradient Method for Solving Linear Systems
Your task is to implement the Conjugate Gradient (CG) method, an efficient iterative
algorithm for solving large, sparse, symmetric, positive-definite linear systems.
Given a matrix A and a vector b, the algorithm will solve for x in the system ( Ax = b ).

Write a function conjugate_gradient(A, b, n, x0=None, tol=1e-8) that performs the
Conjugate Gradient method as follows:
 - A: A symmetric, positive-definite matrix representing the linear system.
 - b: The vector on the right side of the equation.
 - n: Maximum number of iterations.
 - x0: Initial guess for the solution vector.
 - tol: Tolerance for stopping criteria.

The function should return the solution vector x.

Example:
    Input:
        A = np.array([[4, 1], [1, 3]])
        b = np.array([1, 2])
        n = 5

        print(conjugate_gradient(A, b, n))
    Output:
        [0.09090909, 0.63636364]
Reasoning:
    The Conjugate Gradient method is applied to the linear system Ax = b with
    the given matrix A and vector b. The algorithm iteratively refines the solution
    to converge to the exact solution.
"""


import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0

    r = b - A @ x
    p = r

    for _ in range(n):
        alpha = r.T @ r / (p.T @ A @ p)
        x = x + alpha * p
        new_r = r - alpha * (A @ p)
        if np.linalg.norm(new_r) < tol:
            break

        beta = new_r.T @ new_r / (r.T @ r)

        p = new_r + beta * p
        r = new_r

    return x
