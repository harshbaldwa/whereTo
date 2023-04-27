import pandas as pd

from .filters import filter_by_checkins, group_checkins


def load_data(dataset, min_checkins=5, max_checkins=50):
    if dataset == "gowalla-small":
        df_checkins = pd.read_csv(
            "data/gowalla-small/checkins.txt", sep="\t", header=None)
        df_checkins.columns = [
            "user_id", "checkin_time", "latitude", "longitude", "location_id"]
    elif dataset == "gowalla":
        df_checkins = pd.read_csv("data/gowalla/gowalla_checkins.txt", sep=",")
        df_checkins.columns = [
            "user_id", "location_id", "checkin_time"]
    elif dataset == "foursquare":
        df_checkins = pd.read_csv(
            "data/foursquare/checkins.txt", sep="\t", header=None)
        df_checkins.columns = [
            "user_id", "location_id", "utc_time", "time_zone"]
        # remap "utc_time" to "checkin_time"
        df_checkins["checkin_time"] = df_checkins["utc_time"]
    else:
        raise ValueError("Invalid dataset")

    df_checkins = df_checkins[["user_id", "location_id", "checkin_time"]]
    df_checkins = df_checkins.sort_values(by=["user_id"])

    df_checkins = filter_by_checkins(df_checkins, min_checkins, max_checkins)
    df_checkins = group_checkins(df_checkins)

    return df_checkins
