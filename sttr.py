import argparse
import time

import torch.utils.data as data
from STTR.load import *
from STTR.models import *
from STTR.preprocess import *
from STTR.train import *
from torch import optim
from tqdm import tqdm

# Update to change the dataset
dname = 'Foursquare'

parser = argparse.ArgumentParser()
parser.add_argument('--preprocess', type=bool, default=False, help='Preprocess the data')
args = parser.parse_args()

if args.preprocess:
    # Generate the cleaned npy files from the raw data
    preprocess(dname)

    # Generate the initial embeddings as pickle file from above npy files
    create_pickle(dname)

# Train the model with the hyper-parameters
part = 10
emb_dim = 256
dropout = 0
num_neg = 10
lr = 3e-3
epochs = 2

file = open('./data/sttr_files/' + dname + '_data.pkl', 'rb') 
file_data = joblib.load(file)
[trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max] = file_data
mat1, mat2s, mat2t, lens = torch.FloatTensor(mat1), torch.FloatTensor(mat2s).to(device), torch.FloatTensor(mat2t), torch.LongTensor(lens)

trajs, mat1, mat2t, labels, lens = trajs[:part], mat1[:part], mat2t[:part], labels[:part], lens[:part]
ex = mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min()
stan = Model(t_dim=hours+1, l_dim=l_max+1, u_dim=u_max+1, embed_dim=emb_dim, ex=ex, dropout=dropout)

num_params = 0
for param in stan.parameters():
    num_params += param.numel()

records = {'epoch': [], 'recall_valid': [], 'recall_test': [], 'ndcg_valid': [], 'ndcg_test': []}
start = time.time()
load = False
trainer = Trainer(stan, records, load, trajs, mat1, mat2t, labels, lens, mat2s, num_neg, lr, epochs)
trainer.train(part, start, dname, emb_dim)