import pandas as pd


def load_data(dataset, compress_same_ul=False):
    if dataset == "gowalla-small":
        df_checkins = pd.read_csv(
            "data/gowalla-small/checkins.txt", sep="\t", header=None)
        df_friends = pd.read_csv(
            "data/gowalla-small/friends.txt", sep="\t", header=None)

        df_checkins.columns = [
            "user_id", "checkin_time", "latitude", "longitude", "location_id"]
        df_friends.columns = ["user_id", "friend_id"]
    elif dataset == "gowalla":
        # header present in the file (not in the documentation)
        df_checkins = pd.read_csv("data/gowalla/gowalla_checkins.txt", sep=",")
        df_friends = pd.read_csv("data/gowalla/gowalla_friends.txt", sep=",")

        df_checkins.columns = [
            "user_id", "location_id", "checkin_time"]
        df_friends.columns = ["user_id", "friend_id"]
    elif dataset == "foursquare":
        df_checkins = pd.read_csv(
            "data/foursquare/checkins.txt", sep="\t", header=None)
        df_friends = pd.read_csv(
            "data/foursquare/friends.txt", sep="\t", header=None)

        df_checkins.columns = [
            "user_id", "location_id", "utc_time", "time_zone"]
        df_friends.columns = ["user_id", "friend_id"]

    df_checkins = df_checkins[["user_id", "location_id"]]
    df_checkins = df_checkins.sort_values(by=["user_id"])
    if compress_same_ul:
        df_checkins['frequency'] = 1
        df_checkins = df_checkins.groupby(
            ['user_id', 'location_id']).count().reset_index()

    df_friends = df_friends.sort_values(by=["user_id"])
    return df_checkins, df_friends
