import pickle
import time
import warnings

import numpy as np
import torch
from scipy.sparse import csr_matrix, diags, load_npz, save_npz, linalg
from sparsesvd import sparsesvd
from sklearn.decomposition import TruncatedSVD
from torchdiffeq import odeint

from .load_data import load_data, load_data_train_test


warnings.filterwarnings("ignore", message="divide by zero encountered in power")


class BSPM:
    def __init__(
        self, dataset: str, k: int = 100, idl: float = 0.2,
        train_size: float = 0.7, train_seed: int = 42, min_checkins: int = 5,
        max_checkins: int = 25, topk: int = 20
    ):
        self.dataset = dataset
        self.dataset_pre = dataset + "/_pre"
        self.k = k
        self.idl = idl
        self.train_size = train_size
        self.train_seed = train_seed
        self.topk = topk

        self.ground_truth = None
        self.train_data = None
        self.test_data = None
        self.adj_matrix = None

        self.r_prime = None
        self.processed = None
        self.V_inv = None
        self.V = None
        self.vt = None

        self.sharp_out = None
        self.results = None

        filename = f"data/{self.dataset_pre}/adj_list.pkl"
        try:
            with open(filename, "rb") as f:
                self.ground_truth = pickle.load(f)
            with open(f"data/{self.dataset_pre}/data_info.txt", "r") as f:
                self.n_usr, self.n_loc = map(int, f.read().split(", "))
            print("Loaded adj list from file")
        except FileNotFoundError:
            df_checkins = load_data(self.dataset, min_checkins, max_checkins)
            print("Loaded data")
            df_checkins, self.n_usr, self.n_loc = self.remap_ids(df_checkins)
            self.calc_adj_list(df_checkins)
            with open(filename, "wb") as f:
                pickle.dump(self.ground_truth, f)
            with open(f"data/{self.dataset_pre}/data_info.txt", "w") as f:
                f.write(f"{self.n_usr}, {self.n_loc}")
            print("Created adj list")

        print("Users:", self.n_usr)
        print("Locations:", self.n_loc)

        train_p = str(int(self.train_size*100)) + "_" + str(self.train_seed)
        filename = f"data/{self.dataset_pre}/train_{train_p}.pkl"
        filename2 = f"data/{self.dataset_pre}/test_{train_p}.pkl"
        try:
            with open(filename, "rb") as f:
                self.train_data = pickle.load(f)
            with open(filename2, "rb") as f:
                self.test_data = pickle.load(f)
            print("Loaded train data from file")
        except FileNotFoundError:
            self.train_split()
            with open(filename, "wb") as f:
                pickle.dump(self.train_data, f)
            with open(filename2, "wb") as f:
                pickle.dump(self.test_data, f)
            print("Created train data split")

        # self.train_data, self.test_data, self.ground_truth, self.n_usr, self.n_loc = load_data_train_test()
        # self.n_usr += 1
        # self.n_loc += 1

        filename = f"data/{self.dataset_pre}/adj_matrix_{train_p}.npz"
        try:
            adj_matrix = load_npz(filename)
            print("Loaded adj matrix from file")
        except FileNotFoundError:
            adj_matrix = self.make_adj_matrix()
            save_npz(filename, adj_matrix)
            print("Created adj matrix")

        self.adj_matrix = adj_matrix.tolil()
        # self.print_train_test()
        # exit()
        self.calc_factors()

        filename = f"data/{self.dataset_pre}/svd_{self.k}.npy"
        try:
            self.vt = np.load(filename)
            print("Loaded SVD from file")
        except FileNotFoundError:
            self.do_svd()
            np.save(filename, self.vt)
            print("Created SVD")

    def remap_ids(self, df_checkins):
        # get unique user ids and locations ids
        users = df_checkins["user_id"].unique()
        locations = df_checkins["location_id"].unique()

        # create a mapping from old ids to new ids
        user_map = dict(zip(users, range(len(users))))
        location_map = dict(zip(locations, range(len(locations))))

        # remap user ids and location ids
        df_checkins["user_id"] = df_checkins["user_id"].apply(
            lambda x: user_map[x]
        )
        df_checkins["location_id"] = df_checkins["location_id"].apply(
            lambda x: location_map[x]
        )

        return df_checkins, len(users), len(locations)

    def calc_adj_list(self, df_checkins):
        # get unique user ids and locations ids
        users = df_checkins["user_id"].unique()

        # create empty adjacency list
        adj_list = {user: [] for user in users}

        # populate adjacency list
        for _, row in df_checkins.iterrows():
            adj_list[row["user_id"]].append(
                # (row["location_id"], row["frequency"], row["last_visited"])
                row["location_id"]
            )

        self.ground_truth = adj_list

    def train_split(self):
        # set random seed
        np.random.seed(self.train_seed)

        # create empty train set
        train_set = {user: [] for user in self.ground_truth}
        test_set = {user: [] for user in self.ground_truth}

        # populate train set
        for user in self.ground_truth:
            n_locs = len(self.ground_truth[user])
            n_train = int(np.ceil(n_locs * self.train_size))
            train_set[user] = np.random.choice(
                self.ground_truth[user], n_train, replace=False
            )
            test_set[user] = list(
                set(self.ground_truth[user]) - set(train_set[user])
            )

        self.train_data = train_set
        self.test_data = test_set

    def make_adj_matrix(self):
        train_u = []
        train_l = []
        for user in self.train_data:
            train_u.extend([user] * len(self.train_data[user]))
            train_l.extend(self.train_data[user])

        train_u = np.array(train_u)
        train_l = np.array(train_l)
        train_freq = np.ones(len(train_u))

        # create adjacency matrix
        adj_matrix = csr_matrix(
            (train_freq, (train_u, train_l)), shape=(self.n_usr, self.n_loc)
        )

        return adj_matrix

    def calc_factors(self):
        adj_matrix = self.adj_matrix
        print("Calculating factors")
        u_diag = np.array(adj_matrix.sum(axis=1))
        u_inv = np.power(u_diag, -0.5).flatten()
        u_inv[np.isinf(u_inv)] = 0
        U_inv = diags(u_inv)
        r_prime = U_inv.dot(adj_matrix)

        v_diag = np.array(adj_matrix.sum(axis=0))
        v_inv = np.power(v_diag, -0.5).flatten()
        v_inv[np.isinf(v_inv)] = 0
        V_inv = diags(v_inv)
        self.V_inv = V_inv
        self.V = diags(v_diag.squeeze())
        r_prime = r_prime.dot(V_inv)
        self.r_prime = r_prime.tocsc()
        self.processed = r_prime.T @ r_prime

    def do_svd(self):
        # _, ss, self.vt = linalg.svds(self.r_prime, self.k, return_singular_vectors="vh")
        # ut, s, self.vt = sparsesvd(self.r_prime, self.k)
        svd = TruncatedSVD(n_components=self.k, n_iter=5, random_state=42)
        svd.fit(self.r_prime)
        self.vt = svd.components_

    def blur_function(self, t, r):
        out = r.numpy() @ self.processed
        out -= r.numpy()
        return torch.Tensor(out)

    def idl_function(self, t, r):
        out = r.numpy() @ self.V_inv @ self.vt.T @ self.vt @ self.V
        out -= r.numpy()
        return torch.Tensor(out)

    def sharp_function(self, t, r):
        out = -r.numpy() @ self.processed
        return torch.Tensor(out)

    def do_thing(self, batch_test, tb=1, kb=2, ti=1, ki=2, ts=2.5, ks=2, idl=None):
        if idl is not None:
            self.idl = idl
        R = torch.Tensor(np.array(self.adj_matrix[batch_test].todense()))
        blurred_out = odeint(self.blur_function, R, torch.linspace(0, tb, kb).float(), method="euler")
        # print("Blurred")
        idl_out = odeint(self.idl_function, R, torch.linspace(0, ti, ki).float(), method="euler")
        # print("IDL")
        sharp_out = odeint(
            self.sharp_function, blurred_out[-1] + self.idl*idl_out[-1],
            torch.linspace(0, ts, ks).float(), method="rk4")

        self.sharp_out = sharp_out[-1]

    def calc_recall(self, batch_test):
        # calculate recall
        recall = 0
        real_counts = 0
        for i, user in enumerate(batch_test):
            if len(self.test_data[user]) == 0:
                continue
            # get top k predictions
            top_k = self.sharp_out[i].numpy().argsort()[-self.topk:][::-1]

            # get true locations
            true_locs = self.test_data[user]

            # calculate recall
            recall += len(set(top_k) & set(true_locs)) / len(true_locs)
            real_counts += 1

        self.results = recall / real_counts

    def pprint_results(self):
        print(f"Recall@{self.topk}: {self.results}")

    def print_train_test(self):
        print("Train:")
        with open(f"data/{self.dataset_pre}/train.txt", "w") as f:
            for user in self.train_data:
                f.write(f"{user} {' '.join([str(loc) for loc in self.train_data[user]])}\n")

        print("Test:")
        with open(f"data/{self.dataset_pre}/test.txt", "w") as f:
            for user in self.test_data:
                f.write(f"{user} {' '.join([str(loc) for loc in self.test_data[user]])}\n")
