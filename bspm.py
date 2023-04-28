import argparse
import time

import numpy as np
from bspm.hbspm import BSPM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="gowalla",
        help="Dataset to use: either 'gowalla' or 'foursquare'")
    parser.add_argument(
        "--recall", type=int, default=20,
        help="Recall@k to calculate")
    parser.add_argument(
        "--idl", type=float, default=0.2,
        help="idl (β) factor")
    parser.add_argument(
        "--k", type=int, default=200,
        help="top k SVD factors")
    parser.add_argument(
        "--train_size", type=float, default=0.7,
        help="Train split size")
    parser.add_argument(
        "--train_seed", type=int, default=42,
        help="Train split seed")
    parser.add_argument(
        "--min_checkins", type=int, default=5,
        help="Minimum number of checkins per user")
    parser.add_argument(
        "--max_checkins", type=int, default=25,
        help="Maximum number of checkins per user")

    args = parser.parse_args()

    bspm = BSPM(
        args.dataset, k=args.k, idl=args.idl, train_size=0.7, train_seed=42,
        min_checkins=5, max_checkins=25, topk=args.recall)

    # create a batch of 200 random users for testing
    np.random.seed(45)
    batch_test = np.random.randint(low=0, high=bspm.n_usr, size=200)

    start = time.time()
    # "train" the model
    bspm.do_thing(batch_test, tb=2.0, ti=2.0, idl=args.idl)
    # calculate the recall@k
    bspm.calc_recall(batch_test)
    # save the results in "results/" folder
    bspm.save_results()
    end = time.time()
    print(f"Time taken for generating results: {end - start}")
