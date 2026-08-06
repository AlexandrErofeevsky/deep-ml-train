"""
Optimal String Alignment Distance
 - Medium
 - NLP

In this problem, you need to implement a function that calculates the Optimal
String Alignment (OSA) distance between two given strings. The OSA distance
represents the minimum number of edits required to transform one string into another.
The allowed edit operations are:

 - Insert a character
 - Delete a character
 - Substitute a character
 - Transpose two adjacent characters
 - Each of these operations costs 1 unit.

Your task is to find the minimum number of edits needed to convert the first
string (s1) into the second string (s2).

For example, the OSA distance between the strings caper and acer is 2: one
deletion (removing "p") and one transposition (swapping "a" and "c").

Example:
    Input:
        source = "butterfly"
        target = "dragonfly"

        distance = OSA(source, target)
        print(distance)
    Output:
        6
Reasoning:
    The OSA distance between the strings "butterfly" and "dragonfly" is 6.
    The minimum number of edits required to transform the source string into
    the target string is 6.
"""
import numpy as np
from numpy.matrixlib.defmatrix import matrix


def OSA(source: str, target: str) -> int:
    matrix = np.zeros((len(source) + 1, len(target) + 1))
    matrix[0] = np.arange(len(target) + 1)
    matrix[:, 0] = np.arange(len(source) + 1)

    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            equation_list = [matrix[i - 1, j] + 1, matrix[i, j - 1] + 1]

            equation_list.append(matrix[i - 1, j - 1] + int(source[i - 1] != target[j - 1]))

            if i > 1 and j > 1 and source[i - 1] == target[j - 2] and source[i - 2] == target[j - 1]:
                equation_list.append(matrix[i - 2, j - 2] + 1)

            matrix[i, j] = min(equation_list)
    return matrix[-1, -1]
