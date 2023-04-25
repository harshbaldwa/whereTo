import numpy as np
import time
from scipy.sparse import dok_matrix, diags, save_npz, load_npz, coo_matrix
from sparsesvd import sparsesvd
from torchdiffeq import odeint
import torch
import pickle

from .load_data import load_data


class BSPM:
    def __init__(self, dataset, top_k):
        self.dataset = dataset
        self.top_k = top_k
        self.n_users = 0
        self.n_locations = 0
        self.adj_mtx = None

        self.lst = None
        self.train_lst = None
        self.P = None
        self.V = None
        self.V_inv = None
        self.Ut = None
        self.idl = 0.2

        self.results_batch = {}

    def load_adj_matrix(self):
        """loads an adjacency matrix from a file if it exists, \
            otherwise creates it from scratch

        Args:
            dataset (str): name of dataset
        """
        try:
            self.adj_mtx = load_npz(f"data/{self.dataset}/adj_matrix.npz")
            with open(f"data/{self.dataset}/adj_list.pkl", "rb") as f:
                self.lst = pickle.load(f)
            with open(f"data/{self.dataset}/train_list.pkl", "rb") as f:
                self.train_lst = pickle.load(f)
            self.n_users, self.n_locations = self.adj_mtx.shape
            self.P = load_npz(f"data/{self.dataset}/P.npz")
            self.V = load_npz(f"data/{self.dataset}/V.npz")
            self.V_inv = load_npz(f"data/{self.dataset}/V_inv.npz")
            self.Ut = load_npz(f"data/{self.dataset}/Ut_{self.top_k}.npz")
        except FileNotFoundError:
            df_checkins, _ = load_data(self.dataset, compress_same_ul=True)
            print("Loaded data")
            df_checkins, self.n_users, self.n_locations = self.remap_ids(df_checkins)
            print("Remapped ids")
            self.adj_list(df_checkins)
            print("Created adj list")
            self.train_split()
            print("Created train split")
            # save self.lst to file
            with open(f"data/{self.dataset}/adj_list.pkl", "wb") as f:
                pickle.dump(self.lst, f)
            with open(f"data/{self.dataset}/train_list.pkl", "wb") as f:
                pickle.dump(self.train_lst, f)
            mtrx = self.adj_matrix()
            print("Created adj matrix")
            self.adj_mtx = mtrx.tolil()
            save_npz(f"data/{self.dataset}/adj_matrix.npz", mtrx)
            self.calc_all_things()

    def remap_ids(self, df_checkins):
        # get unique user ids and locations ids
        users = df_checkins["user_id"].unique()
        locations = df_checkins["location_id"].unique()

        # create a mapping from old ids to new ids
        user_map = dict(zip(users, range(len(users))))
        location_map = dict(zip(locations, range(len(locations))))

        # remap user ids and location ids
        df_checkins["user_id"] = df_checkins["user_id"].apply(lambda x: user_map[x])
        df_checkins["location_id"] = df_checkins["location_id"].apply(
            lambda x: location_map[x]
        )

        return df_checkins, len(users), len(locations)

    def adj_list(self, df_checkins):
        # get unique user ids and locations ids
        users = df_checkins["user_id"].unique()

        # create empty adjacency list
        adj_list = {user: [] for user in users}

        # populate adjacency list
        for user, location in zip(df_checkins["user_id"], df_checkins["location_id"]):
            adj_list[user].append(location)

        self.lst = adj_list

    def train_split(self, train_size=0.7, seed=42):
        """splits the adjacency list into train set

        Args:
            adj_list (dict): adjacency list
            train_size (float, optional): proportion of data to use for \
                training. Defaults to 0.7.
            seed (int, optional): random seed. Defaults to 42.

        Returns:
            dict: train set
        """
        # set random seed
        np.random.seed(seed)

        # create empty train and test sets
        train_set = {user: [] for user in self.lst}

        # populate train and test sets
        for user in self.lst:
            n_locations = len(self.lst[user])
            # if n_locations == 20:
            #     print(user)

            if n_locations == 0:
                continue
            n_train = int(np.ceil(n_locations * train_size))
            train_set[user] = np.random.choice(self.lst[user], n_train, replace=False)

        self.train_lst = train_set

    def adj_matrix(self):
        train_users = []
        train_locations = []
        for user in self.train_lst:
            train_users.extend([user] * len(self.train_lst[user]))
            train_locations.extend(self.train_lst[user])

        train_users = np.array(train_users)
        train_locations = np.array(train_locations)
        train_data = np.ones(len(train_users))

        mtrx = coo_matrix(
            (train_data, (train_users, train_locations)),
            shape=(self.n_users, self.n_locations),
        )
        return mtrx

    def calc_all_things(self):
        """calculates P from the adjacency matrix

        Args:
            adj_matrix (scipy.sparse.dok_matrix): adjacency matrix
        """
        # # row and column sums
        # u_diag = np.array(self.adj_mtx.sum(axis=1))
        # v_diag = np.array(self.adj_mtx.sum(axis=0))
        # print("diag done")
        # # inverse of row and column sums
        # u_inv = np.power(u_diag, -0.5).flatten()
        # u_inv[np.isinf(u_inv)] = 0
        # v_inv = np.power(v_diag, -0.5).flatten()
        # v_inv[np.isinf(v_inv)] = 0
        # print("inv done")
        # # creating sparse diagonal matrices
        # U_inv = diags(u_inv)
        # v_diag = np.power(v_inv, -1).flatten()
        # v_diag[np.isinf(v_diag)] = 0
        # self.V = diags(v_diag)
        # self.V_inv = diags(v_inv)
        # print("diag mtrx done")
        # # calculating R_prime
        # R_prime = U_inv.dot(self.adj_mtx).dot(self.V_inv)
        # self.P = R_prime.T @ R_prime
        # print("R_prime done")
        # R_prime = R_prime.tocsc()
        # print("R_prime csc done")
        # # calculating U_prime (top_k singular vectors)
        # # _, _, self.Ut = linalg.svds(R_prime, k=self.top_k)
        # _, _, self.Ut = sparsesvd(R_prime, self.top_k)
        # self.Ut = csc_matrix(self.Ut)
        # print("done")
        # save_npz(f"data/{self.dataset}/P.npz", self.P)
        # save_npz(f"data/{self.dataset}/V.npz", self.V)
        # save_npz(f"data/{self.dataset}/V_inv.npz", self.V_inv)
        # save_npz(f"data/{self.dataset}/Ut_{self.top_k}.npz", self.Ut)
        adj_mtx = self.adj_mtx
        start = time.time()
        u_diag = np.array(adj_mtx.sum(axis=1))
        u_inv = np.power(u_diag, -0.5).flatten()
        u_inv[np.isinf(u_inv)] = 0.0
        U_inv = diags(u_inv)
        R_prime = U_inv.dot(adj_mtx)

        v_diag = np.array(adj_mtx.sum(axis=0))
        v_inv = np.power(v_diag, -0.5).flatten()
        v_inv[np.isinf(v_inv)] = 0.0
        V_inv = diags(v_inv)
        self.V_inv = V_inv
        self.V = diags(1 / v_inv)
        R_prime = R_prime.dot(V_inv)
        R_prime = R_prime.tocsc()
        self.P = R_prime.T @ R_prime
        print("P done")
        harsh = time.time()
        ut, s, self.vt = sparsesvd(R_prime, self.top_k)
        end = time.time()
        print(f"harsh: {harsh-start}, svd: {end-harsh}")

    def blur_function(self, t, R):
        out = R.numpy() @ self.P
        out -= R.numpy()
        return torch.Tensor(out)

    def idl_function(self, t, R):
        out = R.numpy() @ self.V_inv @ self.Ut.T @ self.Ut @ self.V
        out -= R.numpy()
        return torch.Tensor(out)

    def sharp_function(self, t, R):
        out = -R.numpy() @ self.P
        return torch.Tensor(out)

    def train(self, batch_test):

        self.results_batch = {user: [] for user in batch_test}
        print("data loaded")
        R = torch.Tensor(np.array(self.adj_mtx[batch_test].todense()))
        blurred_out = odeint(
            self.blur_function, R, torch.linspace(0, 1, 2).float(), method="euler"
        )
        print("blurred")
        idl_out = odeint(
            self.idl_function, R, torch.linspace(0, 1, 2).float(), method="euler"
        )
        print("idl")
        sharp_out = odeint(
            self.sharp_function,
            blurred_out[-1] + self.idl * idl_out[-1],
            torch.linspace(0, 2.5, 2).float(),
            method="rk4",
        )
        print("sharp")
        for i, user in enumerate(batch_test):
            self.results_batch[user] = sharp_out[-1][i].numpy().argsort()[::-1]
        print("done")

    def predict(self, user, k=20):
        return self.results_batch[user][:k]
