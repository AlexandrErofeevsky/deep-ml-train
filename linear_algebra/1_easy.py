def matrix_dot_vector(
    a: list[list[int | float]],
    b: list[int | float],
) -> list[int | float] | int:
    """
    Return a list where each element is the dot product of a row of 'a' with 'b'.
    If the number of columns in 'a' does not match the length of 'b', return -1.
    """
    if len(a[0]) != len(b):
        return -1

    return [
        sum([el_mat * el_vec for el_mat, el_vec in zip(col, b)]) for col in a
    ]
