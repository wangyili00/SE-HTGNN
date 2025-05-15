import os.path as osp
import torch
from torch_geometric.transforms import ToUndirected
from collections import Counter
from torch_geometric.data import HeteroData, Data
import numpy as np
from torch_geometric.utils import negative_sampling
import dgl
from data.process import AminerDataset

def time_merge_edge_time(datalist):
    """merge dataset selected by time. time is stored in edge_attr in Long Tensor
    @params datalist : list of HeteroData
    @return : merged HeteroData
    """
    d = datalist
    d0 = d[0]
    dt = HeteroData()
    for ntype, value in d0.x_dict.items():
        dt[ntype].x = value
    edge_index_dict = {}
    edge_time_dict = {}
    for etype in d0.edge_time_dict:
        for i in range(len(d)):
            edge_index_dict[etype] = edge_index_dict.get(etype, [])
            edge_index_dict[etype].append(d[i].edge_index_dict[etype])
            edge_time_dict[etype] = edge_time_dict.get(etype, [])
            edge_time_dict[etype].append(d[i].edge_time_dict[etype])
        dt[etype].edge_index = torch.cat(edge_index_dict[etype], dim=1)
        dt[etype].edge_time = torch.cat(edge_time_dict[etype], dim=0)
    return dt

def hetero_linksplit(data, etype, device, inplace=False):
    """hetero_data, do negative sampling
    @params data: hetero_data
    @return : hetero_data
    """
    if not inplace:
        data = data.clone()
    # device = data[etype].edge_index.device
    ei = data[etype].edge_index.to("cpu").numpy()

    # reorder
    nodes0 = list(set(ei[0].flatten()))
    nodes0.sort()
    nodes1 = list(set(ei[1].flatten()))
    nodes1.sort()

    id2n0 = nodes0
    id2n1 = nodes1
    n02id = dict(zip(nodes0, np.arange(len(nodes0))))
    n12id = dict(zip(nodes1, np.arange(len(nodes1))))
    size = (len(nodes0), len(nodes1))
    # construct the graph containing these nodes
    ei_ = np.apply_along_axis(lambda x: (n02id[x[0]], n12id[x[1]]), axis=0, arr=ei)
    nei_ = negative_sampling(torch.LongTensor(ei_), size).numpy()
    nei = torch.LongTensor(
        np.apply_along_axis(lambda x: (id2n0[x[0]], id2n1[x[1]]), axis=0, arr=nei_)
    )
    ei = torch.LongTensor(ei)

    # add to edge attr
    data[etype].edge_label_index = torch.cat([ei, nei], dim=-1).to(device)
    data[etype].edge_label = torch.cat(
        [torch.ones(ei.shape[1]), torch.zeros(nei.shape[1])], dim=-1
    ).to(device)
    pos_hetero_dict = {('user', 'interact', 'item'): (ei[0],ei[1])}
    neg_hetero_dict = {('user', 'interact', 'item'): (nei[0],nei[1])}
    num_nodes_dict = {'item': 34505, 'user': 1476}
    return (dgl.heterograph(pos_hetero_dict, num_nodes_dict).to(device),
            dgl.heterograph(neg_hetero_dict, num_nodes_dict).to(device))


def linksplit(data, device, all_neg=False, inplace=False, num_nodes = 0):
    """HomoData, do negative sampling
    @params data: HomoData
    @return : HomoData
    """
    if not inplace:
        data = data.clone()
    # device = data.edge_index.device
    ei = data.edge_index.to("cpu").numpy()

    # reorder
    nodes = list(set(ei.flatten()))
    nodes.sort()
    id2n = nodes
    n2id = dict(zip(nodes, np.arange(len(nodes))))

    # construct the graph containing these nodes
    ei_ = np.vectorize(lambda x: n2id[x])(ei)

    if all_neg:
        maxn = len(nodes)
        nei_ = []
        pos_e = set([tuple(x) for x in ei_.T])
        for i in range(maxn):
            for j in range(maxn):
                if i != j and (i, j) not in pos_e:
                    nei_.append([i, j])
        nei_ = torch.LongTensor(nei_).T
    else:
        nei_ = negative_sampling(torch.LongTensor(ei_))
    nei = torch.LongTensor(np.vectorize(lambda x: id2n[x])(nei_.numpy()))
    ei = torch.LongTensor(ei)

    # add to edge attr
    data.edge_label_index = torch.cat([ei, nei], dim=-1).to(device)
    data.edge_label = torch.cat([torch.ones(ei.shape[1]), torch.zeros(nei.shape[1])], dim=-1).to(
        device
    )
    # pos neg
    return (dgl.graph((ei[0],ei[1]),num_nodes=num_nodes).to(device), dgl.graph((nei[0],nei[1]),num_nodes=num_nodes).to(device))
    # return data


def time_merge(glist, num_nodes_dict, link_pre=True):

    hetero_dict = {}
    for (t, g_s) in enumerate(glist):
        for srctype, etype, dsttype in g_s.edge_types:
            src = g_s[etype].edge_index[0]
            dst = g_s[etype].edge_index[1]
            hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (src, dst)
            hetero_dict[(dsttype, f'{etype}_r_t{t}', srctype)] = (dst, src)
    G_feat = dgl.heterograph(hetero_dict, num_nodes_dict)

    # for ntype in G_feat.ntypes:
    #     G_feat[ntype].x = glist[0][ntype].x

    # G_feat = dgl.heterograph()
    for (t, g_s) in enumerate(glist):
        for ntype in G_feat.ntypes:
            # G_feat.nodes[ntype].data[f't{t}'] = torch.zeros(G_feat.num_nodes(ntype), g_s.nodes[ntype].data['feat'].shape[1])  # 第一维为大图的所有节点，第二维为所有特诊的维度，意味着该时刻不存在的节点的特征补0
            if ntype in ['user','item',"author",] and link_pre: #'author',"venue"
                # G_feat.nodes[ntype].data[f't{t}'] = torch.empty(g_s[ntype].num_nodes, 32, dtype=torch.float32).uniform_(-1.0, 1.0)
                G_feat.nodes[ntype].data[f't{t}'] = g_s[ntype].x.type(torch.float32).unsqueeze(1)
            else:
                G_feat.nodes[ntype].data[f't{t}'] = g_s[ntype].x
    # g_s[ntype].num_nodes
    return G_feat

def remove_edges_unseen_nodes(data, train_nodes):
    """inplace operation, remove edges with nodes not in train_nodes"""
    idxs = []
    ei = data.edge_index.T  # [E,2]
    # print(f'before removing : {ei.T.shape}')
    for i in range(ei.shape[0]):
        e = ei[i].numpy()
        if (e[0] in train_nodes) and (e[1] in train_nodes):
            idxs.append(i)
    idxs = torch.LongTensor(idxs)
    data.edge_index = torch.index_select(data.edge_index, 1, idxs)

def time_select_edge_time(dataset, t):
    """select by time t. time is stored in edge_attr in Long Tensor
    @param t: time index
    @return : HeteroData , with same nodes ,with sliced edgeindex and edge attr
    """
    dt = HeteroData()
    d = dataset
    for ntype, value in d.x_dict.items():
        dt[ntype].x = value
        if "num_nodes" in d[ntype].keys():
            dt[ntype].num_nodes = d[ntype].num_nodes
    dea = d.edge_time_dict
    dei = d.edge_index_dict
    for etype in dea:
        mask = (dea[etype] == t).squeeze(-1)
        dt[etype].edge_index = dei[etype][:, mask]
        dt[etype].edge_time = dea[etype][mask]
    return dt

def get_eval_data(data):
    eval_data = HeteroData()
    # for x in ['user','item']:
    #     eval_data[x]=data[x]
    eval_data[tuple("user interact item".split())].edge_index = data[
        "interact"
    ].edge_index
    return eval_data

def get_author_graph(data):
    """get coauthor graph
    @params data : HeteroData
    @return coauthor graph: pyg.Data
    """
    es = data["written"].edge_index.numpy().T

    # e1 is paper , e2 is author
    p2a = {}
    for (e1, e2) in es:
        if e1 not in p2a:
            p2a[e1] = set()
        p2a[e1].add(e2)

    # author graph does not include self-edge
    author_edges = []
    for p, authors in p2a.items():
        authors = list(authors)
        na = len(authors)
        for i in range(na):
            for j in range(i + 1, na):
                author_edges.append([authors[i], authors[j]])
                author_edges.append([authors[j], authors[i]])
    author_edges = torch.LongTensor(np.array(author_edges).T)
    data = Data(edge_index=author_edges)
    return data

def load_am_data(time_window, device):
    # time_window = 8
    setup_seed(22)
    # processed = "data/Aminer/processed-False-venue32.pt"
    # if osp.exists(processed):
    #     print(f'loading {processed}')
    #     dataset = torch.load(processed)
    # else:
    #     dataset = AminerDataset(undirected=True, word2vec_size=32).dataset
    #     torch.save(dataset, processed)
    dataset = AminerDataset(undirected=True, word2vec_size=32).dataset

    #  preprocess

    years = (
                sorted(list(Counter(dataset.time_dict["paper"].squeeze().numpy()).keys()))
            )  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    for nt in "author venue".split():
        dataset[nt].x = dataset[nt].x.squeeze(-1)
    # dataset['author'].x = dataset['author'].x.squeeze(-1)

    datas = [time_select_edge_time(dataset, i) for i in years]  # heteros
    eval_datas = [get_author_graph(datas[k]) for k in years]  # localidxs
    train_idx, val_idx, test_idx = 13, 14, 15

    # remove edges with nodes not in train_nodes
    train_nodes = [set()]  # [null,edge0,edge1,...]
    for i in range(train_idx):  # [0-12]
        train_i = train_nodes[-1] | set(eval_datas[i].edge_index.unique().numpy())
        train_nodes.append(train_i)

    for i in range(1, train_idx):
        remove_edges_unseen_nodes(
            eval_datas[i], train_nodes[i]
        )  # remove for [1,12]
    for i in range(train_idx, test_idx + 1):
        remove_edges_unseen_nodes(
            eval_datas[i], train_nodes[-1]
        )  # remove for [13-15]

    num_nodes_dict = dataset.num_nodes_dict

    eval_datas = [linksplit(eval_datas[k], device, num_nodes=dataset.num_nodes_dict['author']) for k in years]  # get neg edges


    time_dataset = {}

    time_dataset["train"] = [
        (time_merge([datas[i] for i in range(k - time_window, k)], num_nodes_dict= num_nodes_dict).to(device), eval_datas[k])
        for k in range(time_window, train_idx + 1)  # [0-12] -> [13]
    ]
    time_dataset["val"] = [
        (
            time_merge(
                [datas[i] for i in range(val_idx - time_window,val_idx)],num_nodes_dict= num_nodes_dict
            ).to(device),
            eval_datas[val_idx],
        )
    ]
    time_dataset["test"] = [
        (
             time_merge(
                [datas[i] for i in range(test_idx- time_window, test_idx)],num_nodes_dict= num_nodes_dict
            ).to(device),
            eval_datas[test_idx],
        )
    ]

    train_feats = [time_dataset["train"][i][0] for i in range(len(time_dataset["train"]))]
    train_labels = [time_dataset["train"][i][1] for i in range(len(time_dataset["train"]))]
    val_feats = [time_dataset["val"][i][0] for i in range(len(time_dataset["val"]))]
    val_labels = [time_dataset["val"][i][1] for i in range(len(time_dataset["val"]))]
    test_feats = [time_dataset["test"][i][0] for i in range(len(time_dataset["test"]))]
    test_labels = [time_dataset["test"][i][1] for i in range(len(time_dataset["test"]))]
    return train_feats, train_labels, val_feats, val_labels, test_feats, test_labels


def setorderidx(data):
    """map col idx to ordered idx starting from 0
    @params data: numpy array , rows as samples, columns as attributes
    @return data: numpy array
    """
    data = data.copy()
    row, col = data.shape
    cnt = {}
    for i in range(col):
        cnt[i] = Counter(data[:, i])
        k = list(cnt[i].keys())
        k.sort()
        k2i = dict(zip(k, range(len(k))))
        # print(f'mapping col {i} head 50')
        # print(list(k2i.items())[:50])
        for j in range(row):
            data[j][i] = k2i[data[j][i]]
    data = np.vectorize(int)(data)
    return data


def hetero_remove_edges_unseen_nodes(data, etype, train_nodes0, train_nodes1):
    """inplace operation, remove edges with nodes not in train_nodes"""
    idxs = []
    ei = data[etype].edge_index.T  # [E,2]
    # print(f'before removing : {ei.T.shape}')
    for i in range(ei.shape[0]):
        e = ei[i].numpy()
        if (e[0] in train_nodes0) and (e[1] in train_nodes1):
            idxs.append(i)
    idxs = torch.LongTensor(idxs)
    data[etype].edge_index = torch.index_select(data[etype].edge_index, 1, idxs)


def load_eco_data(time_window=7, device='cuda:0'):
    setup_seed(22)
    processed = "data/ecomm/ecomm_edge_train.txt"
    data = []
    with open(processed, "r") as file:
        for line in file:
            nid1, nid2, etype, t = line.split()
            data.append([nid1, nid2, etype, int(t)])
    data = np.array(data)
    data = setorderidx(
        data
    )
    cnt = {}
    for i in range(4):
        cnt[i] = Counter(data[:, i])

    years = list(cnt[3].keys())
    years.sort()

    node1_num = len(cnt[0])
    node2_num = len(cnt[1])
    etypes = [
        tuple("user click item".split()),
        tuple("user buy item".split()),
        tuple("user cart item".split()),
        tuple("user favorite item".split()),
        tuple("user interact item".split()),
    ]
    dataset = HeteroData()
    for i in range(4):
        dataset[etypes[i]].edge_index = torch.LongTensor(data[data[:, 2] == i][:, :2].T)
        dataset[etypes[i]].edge_time = torch.LongTensor(data[data[:, 2] == i][:, [3]])
    dataset[etypes[-1]].edge_index = torch.cat(
        [dataset[etypes[i]].edge_index for i in range(4)], dim=1
    )
    dataset[etypes[-1]].edge_time = torch.cat(
        [dataset[etypes[i]].edge_time for i in range(4)]
    )

    dataset["user"].x = torch.arange(0, node1_num).unsqueeze(-1)
    dataset["item"].x = torch.arange(0, node2_num).unsqueeze(-1)


    for nt in "user item".split():
        dataset[nt].x = dataset[nt].x.squeeze(-1)

    datas = [time_select_edge_time(dataset, i) for i in years]  # heteros
    eval_datas = [get_eval_data(data) for data in datas]  # homo

    eval_etype = tuple("user interact item".split())

    for i in years:
        del datas[i]["interact"], datas[i]["rev_interact"]
    del dataset["interact"], dataset["rev_interact"]

    train_idx, val_idx, test_idx = 7, 8, 9

    train_nodes0 = [set()]  # [null,edge0,edge1,...]
    train_nodes1 = [set()]  # [null,edge0,edge1,...]
    for i in range(train_idx):
        train_i0 = train_nodes0[-1] | set(
            eval_datas[i][eval_etype].edge_index[0].unique().numpy()
        )
        train_i1 = train_nodes1[-1] | set(
            eval_datas[i][eval_etype].edge_index[1].unique().numpy()
        )
        train_nodes0.append(train_i0)
        train_nodes1.append(train_i1)

    for i in range(1, train_idx):
        hetero_remove_edges_unseen_nodes(
            eval_datas[i], eval_etype, train_nodes0[i], train_nodes1[i]
        )

    for i in range(train_idx, test_idx + 1):
        hetero_remove_edges_unseen_nodes(
            eval_datas[i], eval_etype, train_nodes0[-1], train_nodes1[-1]
        )

    # negative sampling
    eval_datas = [
        hetero_linksplit(eval_datas[k], eval_etype, "cuda:0") for k in years
    ]  # homo
    num_nodes_dict = {'item': 34505, 'user': 1476}
    time_dataset = {}
    time_dataset["train"] = [
        (time_merge([datas[i] for i in range(k - time_window, k)],num_nodes_dict), eval_datas[k])
        for k in range(time_window, train_idx + 1)
    ]
    time_dataset["val"] = [
        (
            time_merge(
                [datas[i] for i in range(val_idx - time_window, val_idx)],num_nodes_dict
            ),
            eval_datas[val_idx],
        )
    ]
    time_dataset["test"] = [
        (
            time_merge(
                [datas[i] for i in range(test_idx - time_window, test_idx)],num_nodes_dict
            ),
            eval_datas[test_idx],
        )
    ]

    train_feats = [time_dataset["train"][i][0] for i in range(len(time_dataset["train"]))]
    train_labels = [time_dataset["train"][i][1] for i in range(len(time_dataset["train"]))]
    val_feats = [time_dataset["val"][i][0] for i in range(len(time_dataset["val"]))]
    val_labels = [time_dataset["val"][i][1] for i in range(len(time_dataset["val"]))]
    test_feats = [time_dataset["test"][i][0] for i in range(len(time_dataset["test"]))]
    test_labels = [time_dataset["test"][i][1] for i in range(len(time_dataset["test"]))]

    return train_feats, train_labels, val_feats, val_labels, test_feats, test_labels


def load_yelp_data(time_window=12, device='cuda:0',val_ratio=0.1, test_ratio=0.1,seed = 0):
    setup_seed(seed)  # seed preprocess
    dataset = None
    processed ='data/yelp/processed/True-32.pt'
    if osp.exists(processed):
        # print(f'loading {processed}')
        dataset = torch.load(processed)

    times = sorted(
            list(Counter(dataset["review"].edge_time.squeeze().numpy()).keys())
        )

    datas = [time_select_edge_time(dataset, i) for i in times]  # heteros

    def get_eval_data(dataset, mask):
        # eval_data = Data()
        # eval_data.y = dataset["item"].y
        # eval_data.mask = mask

        ei = torch.LongTensor(([],[]))
        hetero_dict = {('user', 'interact', 'item'): (ei[0], ei[1])}
        num_nodes_dict = {'user': 55702, 'item': 12524, }
        eval_data = dgl.heterograph(hetero_dict, num_nodes_dict=num_nodes_dict)
        eval_data.nodes['item'].data['y'] = dataset["item"].y
        eval_data.nodes['item'].data['mask'] = mask
        return eval_data.to(device)

    train_mask, val_mask, test_mask = train_val_test_split(
        len(dataset["item"].x), val_ratio=val_ratio, test_ratio=test_ratio
    )
    train_eval = get_eval_data(dataset, train_mask)
    val_eval = get_eval_data(dataset, val_mask)
    test_eval = get_eval_data(dataset, test_mask)
    maxn = 12
    patchlen = maxn // time_window
    num_nodes_dict = {'user': 55702, 'item': 12524,}


    # time_dataset = [
    #         time_merge([datas[i] for i in range(0, maxn)], num_nodes_dict, False)
    #     ]
    time_dataset = [
            time_merge([datas[i] for i in range(0, time_window)], num_nodes_dict, False)
        ]
    return time_dataset, [train_eval], time_dataset,[val_eval], time_dataset, [test_eval]


def train_val_test_split(maxn, val_ratio=0.1, test_ratio=0.1):
    val_num = int(np.ceil(val_ratio * maxn))
    test_num = int(np.ceil(test_ratio * maxn))
    train_num = maxn - val_num - test_num
    assert train_num >= 0 and test_num >= 0 and val_num >= 0

    idxs = np.arange(maxn)
    np.random.shuffle(idxs)
    test_idxs = idxs[:test_num]
    val_idxs = idxs[test_num : test_num + val_num]
    train_idxs = idxs[test_num + val_num :]
    print(f"split sizes: train {train_num} ; val {val_num} ; test {test_num}")

    train_mask = torch.zeros(maxn).bool()
    train_mask[train_idxs] = True
    val_mask = torch.zeros(maxn).bool()
    val_mask[val_idxs] = True
    test_mask = torch.zeros(maxn).bool()
    test_mask[test_idxs] = True

    return train_mask, val_mask, test_mask


def setup_seed(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    time_window = 8
    # device = torch.device('cuda:0')
    # # load_am_data(time_window, device)
    # # load_eco_data(time_window, device)
    # load_yelp_data(time_window)
    # src = torch.tensor([1, 1])
    # dst = torch.tensor([2, 3])
    # hetero_dict = {}
    # hetero_dict[('hunter','prey_on','prey')] = (src, dst)
    # G = dgl.heterograph(hetero_dict)
    # print(G)