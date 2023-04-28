import time

import numpy as np
from bspm.hbspm import BSPM

# dataset: either "gowalla" or "foursquare"
dataset = "gowalla"

# Recall@k
recall = 20

# idl (β) factor
idl = 0.2

# top k SVD factors
k = 200

# create the BSPM object
bspm = BSPM(
    dataset, k=k, idl=idl, train_size=0.7, train_seed=42,
    min_checkins=5, max_checkins=30, topk=recall)

# create a batch of 200 random users for testing
np.random.seed(45)
batch_test = np.random.randint(low=0, high=bspm.n_usr, dtype=np.int32)

start = time.time()
# "train" the model
bspm.do_thing(batch_test, tb=2.0, ti=2.0, idl=idl)
end = time.time()
print(f"Time taken for training: {end - start}")
# calculate the recall@k
bspm.calc_recall(batch_test)
# save the results in "results/" folder
bspm.save_results()
