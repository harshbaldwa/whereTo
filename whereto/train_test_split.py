import numpy as np
from scipy.sparse import dok_matrix


def train_test_split(adj_matrix, n_users, n_locations, test_size=0.3, seed=42):
    """splits an adjacency matrix into train and test sets
    Args:
        adj_list (dict): adjacency matrix
        test_size (float): proportion of checkins to be used for testing
    """
    np.random.seed(seed)
    # create empty train and test sets
    train_set = {user: [] for user in range(n_users)}
    test_set = {user: [] for user in range(n_users)}
    for user in range(n_users):
        # get number of checkins for user
        n_checkins = adj_matrix[user, :].sum(axis=1)
        # get number of checkins to be used for testing
        n_test = int(test_size * n_checkins)
        # print((adj_matrix[user, :] == 1).todense().tolist()[0])
        # get indices of checkins to be used for testing
        list_interactions = np.arange(n_locations)[(adj_matrix[user, :] == 1).todense().tolist()[0]]
        test_idx = np.random.choice(
            list_interactions, size=n_test, replace=False)
        # get indices of checkins to be used for training
        train_idx = np.setdiff1d(list_interactions, test_idx)
        # populate train and test sets
        train_set[user] = [idx for idx in train_idx]
        test_set[user] = [idx for idx in test_idx]
        print(f"User {user} has {n_checkins} checkins, {n_test} test checkins, {len(train_idx)} train checkins")
    return train_set, test_set


def adj_matrix(adj_list, n_users, n_locations):
    # create empty adjacency matrix
    adj_matrix = dok_matrix((n_users, n_locations), dtype=np.float32)

    # populate adjacency matrix
    for user in adj_list:
        for location in adj_list[user]:
            adj_matrix[user, location] = 1

    return adj_matrix
