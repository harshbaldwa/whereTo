def stats_dataset(df_checkins, df_friends, verbose=False):
    n_users = df_checkins["user_id"].nunique()
    n_checkins = df_checkins.shape[0]
    n_locations = df_checkins["location_id"].nunique()
    if verbose:
        print("Number of users: ", n_users)
        print("Number of locations: ", n_locations)
        print("Number of checkins: ", n_checkins)
        print("Number of friendships: ", df_friends.shape[0] // 2)
        print()
    return n_users, n_checkins, n_locations


def stats_user(df_checkins, df_friends, verbose=False):
    checkin_count = df_checkins.groupby("user_id").count()

    user_max_checkins = checkin_count.idxmax().values[0]
    max_checkins = checkin_count.max().values[0]
    user_min_checkins = checkin_count.idxmin().values[0]
    min_checkins = checkin_count.min().values[0]
    mean_checkins = checkin_count.mean().values[0]
    median_checkins = checkin_count.median().values[0]
    std_checkins = checkin_count.std().values[0]

    friend_count = df_friends.groupby("user_id").count()

    user_max_friends = friend_count.idxmax().values[0]
    max_friends = friend_count.max().values[0]
    user_min_friends = friend_count.idxmin().values[0]
    min_friends = friend_count.min().values[0]
    mean_friends = friend_count.mean().values[0]
    median_friends = friend_count.median().values[0]
    std_friends = friend_count.std().values[0]

    if verbose:
        print("Max user checkins: ", max_checkins,
              " by user ", user_max_checkins)
        print("Min user checkins: ", min_checkins,
              " by user ", user_min_checkins)
        print("Mean user checkins: ", mean_checkins)
        print("Median user checkins: ", median_checkins)
        print("Std user checkins: ", std_checkins)

        print("Max friends: ", max_friends, " by user ", user_max_friends)
        print("Min friends: ", min_friends, " by user ", user_min_friends)
        print("Mean friends: ", mean_friends)
        print("Median friends: ", median_friends)
        print("Std friends: ", std_friends)
        print()


def stats_location(df_checkins, verbose=False):
    count = df_checkins.groupby("location_id").count()

    max_checkins = count.max().values[0]
    location_max_checkins = count.idxmax().values[0]
    min_checkins = count.min().values[0]
    location_min_checkins = count.idxmin().values[0]
    mean_checkins = count.mean().values[0]
    md_checkins = count.median().values[0]
    std_checkins = count.std().values[0]

    if verbose:
        print(
            "Max location checkins: ",
            max_checkins,
            " by location ",
            location_max_checkins,
        )
        print(
            "Min location checkins: ",
            min_checkins,
            " by location ",
            location_min_checkins,
        )
        print("Mean location checkins: ", mean_checkins)
        print("Median location checkins: ", md_checkins)
        print("Std location checkins: ", std_checkins)
