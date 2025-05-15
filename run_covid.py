import dgl
from dgl.data.utils import load_graphs
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from model.model_LLM_enhance_linear import SEHTGNN, NodePredictor
from utils.pytorchtools import EarlyStopping
from utils.data import load_COVID_data
# seed = 0

def SetSeed(seed):
    dgl.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SetSeed(0)
# %%
def evaluate(model, val_feats, val_labels,):
    val_mae_list, val_rmse_list = [], []
    model.eval()
    with torch.no_grad():
        for (G_feat, G_label) in zip(val_feats, val_labels):
            h = model[0](G_feat.to(device), 'state',)
            pred = model[1](h)
            label = G_label.nodes['state'].data['feat']
            loss = F.l1_loss(pred, label.to(device))
            rmse = torch.sqrt(F.mse_loss(pred, label.to(device)))

            val_mae_list.append(loss.item())
            val_rmse_list.append(rmse.item())

        loss = sum(val_mae_list) / len(val_mae_list)
        rmse = sum(val_rmse_list) / len(val_rmse_list)

    return loss, rmse

# %%
device = torch.device('cuda:0')
glist, _ = load_graphs('data/Covid19/covid_graphs.bin')
LLM_features = torch.load("data/Covid19/LLM_feature_Llama-3-new.pt")
time_window = 7
train_feats, train_labels, val_feats, val_labels, test_feats, test_labels = load_COVID_data(glist, time_window,testlen=30)
train_feats = train_feats
graph_atom = test_feats[0]
mae_list, rmse_list = [], []
model_out_path = 'output/COVID19'

best = 0
best_epoch = 0
for k in range(1):

    htgnn = SEHTGNN(graph=graph_atom, n_inp=1, n_hid=8, n_layers=1, n_heads=1, time_window=time_window, norm=False, dropout=0.,#8
                  device=device, LLM_feature=LLM_features)
    predictor = NodePredictor(n_inp=8, n_classes=1)
    model = nn.Sequential(htgnn, predictor).to(device)

    print(f'---------------Repeat time: {k + 1}---------------------')
    print(f'# params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    early_stopping = EarlyStopping(patience=5, verbose=True, path=f'{model_out_path}/checkpoint_HTGNN_{k}.pt')
    optim = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4) #lr=5e-3, weight_decay=5e-4

    train_mae_list, train_rmse_list = [], []
    # idx = np.random.permutation(len(train_feats))
    start = time.time()
    for epoch in range(500):
        model.train()
        for i in range(len(train_feats)):
            # if random.random() < 0.1:
            #     continue
            G_feat = train_feats[i]
            G_label = train_labels[i]

            h = model[0](G_feat.to(device), 'state',)
            pred = model[1](h)
            label = G_label.nodes['state'].data['feat']
            loss = F.l1_loss(pred, label.to(device))
            rmse = torch.sqrt(F.mse_loss(pred, label.to(device)))

            train_mae_list.append(loss.item())
            train_rmse_list.append(rmse.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(sum(train_mae_list) / len(train_mae_list), sum(train_rmse_list) / len(train_rmse_list))

        loss, rmse = evaluate(model, val_feats, val_labels,)
        early_stopping(loss, model,epoch,loss)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    end = time.time()
    # print(f'Total time: {end - start}s')
    model.load_state_dict(torch.load(f'{model_out_path}/checkpoint_HTGNN_{k}.pt'))
    mae, rmse = evaluate(model, test_feats, test_labels,)
    print(f'test_mae: {mae}, test_rmse: {rmse}')

