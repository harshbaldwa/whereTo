from whereto.load_data import load_data
from whereto.eda import stats_dataset
from whereto.filters import filter_by_checkins, group_checkins


def test_data_stats_gwla_sml():
    df_checkins, _ = load_data("gowalla-small")
    n_users, n_checkins, n_locations, n_max_checkins = stats_dataset(df_checkins)
    assert n_users == 57875 and n_checkins == 959844 \
        and n_locations == 521491 and n_max_checkins == 50


def test_data_stats_gwla():
    df_checkins, _ = load_data("gowalla")
    n_users, n_checkins, n_locations = stats_dataset(df_checkins)
    assert n_users == 139214 and n_checkins == 1994916 \
        and n_locations == 906221


# TODO: change the values based on new filters
def test_data_stats_fsq():
    df_checkins, _ = load_data("foursquare")
    n_users, n_checkins, n_locations = stats_dataset(df_checkins)
    assert n_users == 1028506 and n_checkins == 26769947 \
        and n_locations == 5099634