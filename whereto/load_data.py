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
    elif dataset == "test":
        df_checkins = pd.read_csv(
            "data/test/checkins.txt", sep="\t", header=None)
        df_checkins.columns = [
            "user_id", "location_id", "checkin_time"]

    df_checkins = df_checkins[["user_id", "location_id", "checkin_time"]]
    df_checkins = df_checkins.sort_values(by=["user_id"])

    df_checkins = filter_by_checkins(df_checkins, min_checkins, max_checkins)
    df_checkins = group_checkins(df_checkins)

    return df_checkins


def load_data_train_test():
    n_usrs = 0
    n_locs = 0
    ground_truth = {}
    # read the file "train.txt" and "test.txt"
    train_data = {}
    with open("data/gowalla-small/_pre/train.txt", "r") as f:
        train = f.read().splitlines()
    for i in range(len(train)):
        # convert string to list of integers
        temp_list = list(map(int, train[i].split(" ")))
        train_data[temp_list[0]] = temp_list[1:]
        ground_truth[temp_list[0]] = temp_list[1:]
        n_usrs = max(n_usrs, temp_list[0])
        n_locs = max(n_locs, max(temp_list[1:]))

    test_data = {}
    with open("data/gowalla-small/_pre/test.txt", "r") as f:
        test = f.read().splitlines()
    for i in range(len(test)):
        # convert string to list of integers
        temp_list = test[i].split(" ")
        try:
            items = [int(x) for x in temp_list[1:] if x != ""]
            m_items = max(items)
        except:
            items = []
            m_items = 0
        uid = int(temp_list[0])
        test_data[uid] = items
        # ground_truth[uid] = ground_truth[uid].extend(items)
        n_usrs = max(n_usrs, uid)
        n_locs = max(n_locs, m_items)

    return train_data, test_data, ground_truth, n_usrs, n_locs
