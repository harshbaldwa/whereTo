from time import time

from whereto.bspm import load_adj_matrix


datasets = ["gowalla", "foursquare", "gowalla-small"]
for dataset in datasets:
    print(f"Dataset: {dataset}")
    s = time()
    mtrx, n_users, n_locations = load_adj_matrix(dataset)
    e = time()
    print(f"Time to calculate adjacency matrix: {e - s:.3f} seconds")
    s = time()
    mtrx, n_users, n_locations = load_adj_matrix(dataset)
    e = time()
    print(f"Time to load adjacency matrix: {e - s:.3f} seconds")
    print()
