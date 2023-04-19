import numpy as np


def manipulation(matrix):
    # find the maximum value in the matrix row
    matrix = matrix.numpy()
    max_value = np.max(matrix, axis=1)
    factor = 1 / max_value
    # multiply each row by the factor
    matrix = matrix * factor[:, None]
    return matrix