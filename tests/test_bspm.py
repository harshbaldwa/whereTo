import pandas as pd
import numpy as np
from math import sqrt

from whereto import bspm


def test_remaps_ids():
    df_checkins = pd.DataFrame(
        {
            'user_id': [1, 1, 1, 2, 2, 3, 4],
            'location_id': [1, 2, 3, 1, 2, 1, 2],
        }
    )
    df_checkins, n_users, n_locations = bspm.remap_ids(df_checkins)
    assert n_users == 4 and n_locations == 3
    assert df_checkins['user_id'].tolist() == [0, 0, 0, 1, 1, 2, 3]
    assert df_checkins['location_id'].tolist() == [0, 1, 2, 0, 1, 0, 1]


def test_adj_list():
    df_checkins = pd.DataFrame(
        {
            'user_id': [0, 0, 0, 1, 1, 2, 3],
            'location_id': [0, 1, 2, 0, 1, 0, 1],
        }
    )
    adj_list = bspm.adj_list(df_checkins)
    assert adj_list == {
        0: [0, 1, 2],
        1: [0, 1],
        2: [0],
        3: [1],
    }


def test_adj_matrix():
    adj_list = {
        0: [0, 1, 2],
        1: [0, 1],
        2: [0],
        3: [1],
    }
    adj_matrix = bspm.adj_matrix(adj_list, 4, 3)
    assert adj_matrix.todense().tolist() == [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]


def test_calc_P():
    adj_matrix = bspm.adj_matrix(
        {
            0: [0, 1, 2],
            1: [0, 1],
            2: [0],
            3: [1],
        },
        4,
        3,
    )
    P, R_prime, V, V_inv, U_prime = bspm.calc_P(adj_matrix, 4, 3, 1)
    exp_P = [
        [11/18, 5/18, sqrt(1/27)],
        [5/18, 11/18, sqrt(1/27)],
        [sqrt(1/27), sqrt(1/27), 1/3],
    ]
    exp_R_prime = [
        [1/3, 1/3, sqrt(1/3)],
        [sqrt(1/6), sqrt(1/6), 0],
        [sqrt(1/3), 0, 0],
        [0, sqrt(1/3), 0],
    ]
    exp_V = [
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 1],
    ]
    exp_V_inv = [
        [sqrt(1/3), 0, 0],
        [0, sqrt(1/3), 0],
        [0, 0, 1],
    ]
    np.testing.assert_array_almost_equal(P.todense().tolist(), exp_P)
    np.testing.assert_array_almost_equal(
        R_prime.todense().tolist(), exp_R_prime)
    np.testing.assert_array_almost_equal(V.todense().tolist(), exp_V)
    np.testing.assert_array_almost_equal(V_inv.todense().tolist(), exp_V_inv)
