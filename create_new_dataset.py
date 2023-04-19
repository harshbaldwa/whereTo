import numpy as np
from scipy.sparse import csr_matrix, save_npz

from whereto.heat_map import plot_heatmap


n_users = 100
n_locations = 100

adj_list = {user: [] for user in range(n_users)}

np.random.seed(42)

for i in range(n_users):
    num_locations = np.random.randint(1, 3)
    adj_list[i] = np.random.choice(
        n_locations, size=num_locations, replace=True)

adj_matrix = np.zeros((n_users, n_locations))

for user in adj_list:
    for location in adj_list[user]:
        adj_matrix[user, location] = 1

plot_heatmap(adj_matrix, "adj_matrix")

adj_matrix = csr_matrix(adj_matrix)
save_npz("data/harsh/adj_matrix.npz", adj_matrix)
