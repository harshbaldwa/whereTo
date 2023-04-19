import time
import numpy as np

from whereto.bspm import BSPM
from whereto.metrics import recall_at_k

s = time.time()
dataset = "gowalla-small"
bspm = BSPM(dataset, top_k=100)
bspm.load_adj_matrix()
rng = np.random.default_rng()
batch_test = sorted(rng.choice(bspm.n_users, size=20, replace=False))
bspm.train(batch_test)

recall = recall_at_k(bspm, batch_test, 20)
print(f"Recall: {recall}")

# print("Loaded data")

# wsu = 20
# uss = 1
# R = torch.Tensor(np.array(bspm.adj_mtx[wsu:wsu+uss].todense()))

# blurred_out = odeint(
#     bspm.blur_function, R,
#     torch.linspace(0, 1, 2).float(), method="euler")

# print("Blurred")

# idl_out = odeint(
#     bspm.idl_function, R,
#     torch.linspace(0, 1, 2).float(), method="euler")

# print("IDL")

# sharp_out = odeint(
#     bspm.sharp_function, blurred_out[-1] + bspm.idl*idl_out[-1],
#     torch.linspace(0, 2.5, 2).float(), method="rk4")

# print("Sharpened")
# e = time.time()
# print(f"Time: {e-s}")

# for i in range(uss):
#     algo_results = sharp_out[-1][i].numpy().argsort()[-20:][::-1]
#     actual_results = bspm.lst[wsu+i]
#     train_set = bspm.train_lst[wsu+i]

#     new_generated = np.setdiff1d(algo_results, train_set)
#     test = np.setdiff1d(actual_results, train_set)

#     intersection = np.intersect1d(new_generated, test)
#     if len(intersection) > 0:
#         print(f"User: {i+wsu}")
#         print(f"Intersection: {intersection}")
