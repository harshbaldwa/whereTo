from whereto.load_data import load_data
from whereto.eda import stats_dataset


def test_data_stats_gwla_sml():
    df_checkins, df_friends = load_data("gowalla-small")
    n_users, n_checkins, n_locations = stats_dataset(df_checkins, df_friends)
    assert n_users == 107092 and n_checkins == 6442892 \
        and n_locations == 1280969


def test_data_stats_gwla():
    df_checkins, df_friends = load_data("gowalla")
    n_users, n_checkins, n_locations = stats_dataset(df_checkins, df_friends)
    assert n_users == 319063 and n_checkins == 36001959 \
        and n_locations == 2844145


def test_data_stats_fsq():
    df_checkins, df_friends = load_data("foursquare")
    n_users, n_checkins, n_locations = stats_dataset(df_checkins, df_friends)
    assert n_users == 1028506 and n_checkins == 26769947 \
        and n_locations == 5099634