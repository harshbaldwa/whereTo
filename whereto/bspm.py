import numpy as np
from scipy.sparse import dok_matrix, spdiags, linalg

from .load_data import load_data


def remap_ids(df_checkins):
    """remaps user ids and locations ids so that they are contiguous integers

    Args:
        df_checkins (dataframe): dataframe containing \
            checkins [user_id, location_id]
    """
    # get unique user ids and locations ids
    users = df_checkins["user_id"].unique()
    locations = df_checkins["location_id"].unique()

    # create a mapping from old ids to new ids
    user_map = dict(zip(users, range(len(users))))
    location_map = dict(zip(locations, range(len(locations))))

    # remap user ids and location ids
    df_checkins["user_id"] = df_checkins["user_id"].apply(
        lambda x: user_map[x])
    df_checkins["location_id"] = df_checkins["location_id"].apply(
        lambda x: location_map[x])

    return df_checkins, len(users), len(locations)


def adj_list(df_checkins):
    """converts a dataframe of checkins to an adjacency list

    Args:
        df_checkins (dataframe): dataframe containing \
            checkins [user_id, location_id]
    """
    # get unique user ids and locations ids
    users = df_checkins["user_id"].unique()

    # create empty adjacency list
    adj_list = {user: [] for user in users}

    # populate adjacency list
    for user, location in zip(df_checkins["user_id"],
                              df_checkins["location_id"]):
        adj_list[user].append(location)

    return adj_list


def adj_matrix(adj_list, n_users, n_locations):
    """converts an adjacency list to an adjacency matrix

    Args:
        adj_list (dict): adjacency list
    """
    # create empty adjacency matrix
    adj_matrix = dok_matrix((n_users, n_locations), dtype=np.float32)

    # populate adjacency matrix
    for user in adj_list:
        for location in adj_list[user]:
            adj_matrix[user, location] = 1

    return adj_matrix


def calc_P(adj_matrix, n_users, n_locations, top_k):
    """calculates P from the adjacency matrix

    Args:
        adj_matrix (scipy.sparse.dok_matrix): adjacency matrix
    """
    # row and column sums
    u_diag = adj_matrix.sum(axis=1).flatten()
    v_diag = adj_matrix.sum(axis=0).flatten()

    # inverse of row and column sums
    u_inv = np.power(u_diag, -0.5)
    v_inv = np.power(v_diag, -0.5)

    # creating sparse diagonal matrices
    U_inv = spdiags(u_inv, 0, n_users, n_users)
    V = spdiags(v_diag, 0, n_locations, n_locations)
    V_inv = spdiags(v_inv, 0, n_locations, n_locations)

    # calculating R_prime
    R_prime = U_inv @ adj_matrix @ V_inv

    # calculating U_prime (top_k singular vectors)
    U_prime = linalg.svds(R_prime, k=top_k)

    return R_prime.T @ R_prime, R_prime, V, V_inv, U_prime


if __name__ == "__main__":
    df_checkins, _ = load_data("gowalla-small", compress_same_ul=True)
    df_checkins, n_users, n_locations = remap_ids(df_checkins)
    adj_list = adj_list(df_checkins)
    adj_matrix = adj_matrix(adj_list, n_users, n_locations)
    P = calc_P(adj_matrix, n_users, n_locations)
