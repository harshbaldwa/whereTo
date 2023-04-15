from argparse import ArgumentParser

from load_data import load_data

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="gowalla-small",
                        choices=["gowalla-small", "gowalla", "foursquare"])
    args = parser.parse_args()
    df_checkins, df_friends = load_data(args.dataset, True)
