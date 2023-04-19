from whereto.load_data import load_data
from whereto.filters import filter_by_checkins, group_checkins
from whereto.eda import stats_dataset

df_checkins, _ = load_data("gowalla")
df_checkins = filter_by_checkins(df_checkins, min_checkins=5, max_checkins=50)
df_checkins = group_checkins(df_checkins)
n_users, n_checkins, n_locations, n_max_checkins = stats_dataset(df_checkins, verbose=True)
