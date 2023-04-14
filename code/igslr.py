import eda
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gowalla-small")
    args = parser.parse_args()
    df_checkins, df_friends = eda.load_data(args.dataset, True)
