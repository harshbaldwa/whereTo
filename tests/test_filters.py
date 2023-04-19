from whereto.load_data import load_data
from whereto.filters import filter_by_checkins, group_checkins


def test_filter_by_checkins():
    df_checkins, _ = load_data("test")
    df_checkins = filter_by_checkins(df_checkins, min_checkins=2, max_checkins=3)
    assert df_checkins.shape == (4, 3)
    assert df_checkins.iloc[0].user_id == 1
    assert df_checkins.iloc[0].location_id == 0
    assert df_checkins.iloc[0].checkin_time == "2010-10-20T23:55:27Z"
    assert df_checkins.iloc[1].user_id == 1
    assert df_checkins.iloc[1].location_id == 0
    assert df_checkins.iloc[1].checkin_time == "2010-10-19T23:55:27Z"
    assert df_checkins.iloc[2].user_id == 3
    assert df_checkins.iloc[2].location_id == 0
    assert df_checkins.iloc[2].checkin_time == "2010-10-21T23:55:27Z"
    assert df_checkins.iloc[3].user_id == 3
    assert df_checkins.iloc[3].location_id == 0
    assert df_checkins.iloc[3].checkin_time == "2010-10-22T23:55:27Z"


def test_group_checkins():
    df_checkins, _ = load_data("test")
    df_checkins = filter_by_checkins(df_checkins, min_checkins=2, max_checkins=3)
    df_checkins = group_checkins(df_checkins)
    assert df_checkins.shape == (2, 4)
    assert df_checkins.iloc[0].user_id == 1
    assert df_checkins.iloc[0].location_id == 0
    assert df_checkins.iloc[0].frequency == 2
    assert df_checkins.iloc[0].last_visited == "2010-10-20T23:55:27Z"
    assert df_checkins.iloc[1].user_id == 3
    assert df_checkins.iloc[1].location_id == 0
    assert df_checkins.iloc[1].frequency == 2
    assert df_checkins.iloc[1].last_visited == "2010-10-22T23:55:27Z"
