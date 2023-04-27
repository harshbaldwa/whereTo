import time

import numpy as np
from bspm.hbspm import BSPM

dataset = "gowalla"
bspm = BSPM(
    dataset, k=200, idl=0.2, train_size=0.7, train_seed=42,
    min_checkins=5, max_checkins=30, topk=20)

# create a 5 different set of 200 random users

np.random.seed(624010)
batch_test = np.random.choice(
    np.arange(bspm.n_usr), size=(1, 200), replace=False)

for idl in [0.1, 0.2, 0.4, 0.5, 0.8]:
    start = time.time()
    bspm.do_thing(batch_test[0], tb=2.0, ti=2.0, idl=idl)
    end = time.time()
    print(f"Time taken {idl}: {end - start}")
    bspm.calc_recall(batch_test[0])
    bspm.save_results()

bspm = BSPM(
    dataset, k=448, idl=0.2, train_size=0.7, train_seed=42,
    min_checkins=5, max_checkins=30, topk=20)

for idl in [0.1, 0.2, 0.4, 0.5, 0.8]:
    start = time.time()
    bspm.do_thing(batch_test[0], tb=2.0, ti=2.0, idl=idl)
    end = time.time()
    print(f"Time taken {idl}: {end - start}")
    bspm.calc_recall(batch_test[0])
    bspm.save_results()

bspm = BSPM(
    dataset, k=600, idl=0.2, train_size=0.7, train_seed=42,
    min_checkins=5, max_checkins=30, topk=20)

for idl in [0.1, 0.2, 0.4, 0.5, 0.8]:
    start = time.time()
    bspm.do_thing(batch_test[0], tb=2.0, ti=2.0, idl=idl)
    end = time.time()
    print(f"Time taken {idl}: {end - start}")
    bspm.calc_recall(batch_test[0])
    bspm.save_results()
