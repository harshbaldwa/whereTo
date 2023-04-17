import numpy as np


def train_test_split(adj_matrix, n_users, n_locations, test_size=0.3, seed=42):
    """splits an adjacency list into train and test sets
    Args:
        adj_list (dict): adjacency list
        test_size (float): proportion of checkins to be used for testing
    """
    np.random.seed(seed)
    # create empty train and test sets
    train_set = {user: [] for user in range(n_users)}
    test_set = {user: [] for user in range(n_users)}
    for user in range(n_users):
        # get number of checkins for user
        n_checkins = sum(adj_matrix[user, :])
        # get number of checkins to be used for testing
        n_test = int(test_size * n_checkins)
        # get indices of checkins to be used for testing
        list_interactions = np.arange(n_locations)[adj_matrix[user, :] == 1]
        test_idx = np.random.choice(list_interactions, size=n_test, replace=False)
        # get indices of checkins to be used for training
        train_idx = np.setdiff1d(list_interactions, test_idx)
        # populate train and test sets
        train_set[user] = [idx for idx in train_idx]
        test_set[user] = [idx for idx in test_idx]
    return train_set, test_set
