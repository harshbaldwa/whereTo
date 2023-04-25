import time

import numpy as np
import torch
from torchdiffeq import odeint
from whereto.hbspm import BSPM

s = time.time()
dataset = "gowalla"
bspm = BSPM(dataset, k=448, idl=0.2, train_size=0.7, train_seed=42, min_checkins=5, max_checkins=30, topk=20)

np.random.seed(42)
batch_test = np.random.randint(0, bspm.n_usr, 200)

bspm.do_thing(batch_test, tb=2.5, ti=2.5, idl=0.5)
bspm.calc_recall(batch_test)
bspm.pprint_results()
