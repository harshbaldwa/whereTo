import numpy as np

from whereto import train_test_split


def test_train_test_split():
    adj_matrix = np.array([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    train_set, test_set = train_test_split.train_test_split(
        adj_matrix, 4, 4, test_size=0.5, seed=42)
    assert train_set == {
        0: [1, 2],
        1: [0],
        2: [0],
        3: [1],
    }
    assert test_set == {
        0: [0],
        1: [1],
        2: [],
        3: [],
    }
