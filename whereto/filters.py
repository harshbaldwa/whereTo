import pandas as pd


def filter_by_checkins(df_checkins, min_checkins=5, max_checkins=50):
    df_grouped_checkins = df_checkins.groupby(['user_id'], as_index=False).count()
    filtered_user_ids = df_grouped_checkins[(df_grouped_checkins.location_id >= min_checkins) & (df_grouped_checkins.location_id <= max_checkins)].user_id.values
    df_checkins = df_checkins[df_checkins.user_id.isin(filtered_user_ids)]
    return df_checkins


def group_checkins(df_checkins):
    df_grouped_checkins = df_checkins.groupby(['user_id', 'location_id'], as_index=False).agg({'checkin_time': ['count', 'max']})
    df_grouped_checkins.columns = ['user_id', 'location_id', 'frequency', 'last_visited']
    df_grouped_checkins = df_grouped_checkins.sort_values(by=["user_id"])
    df_checkins = df_grouped_checkins
    return df_checkins
