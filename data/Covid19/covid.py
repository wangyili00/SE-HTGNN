import torch
from torch_geometric.data import HeteroData



from dgl.data.utils import load_graphs

glist, _ = load_graphs('covid_graphs.bin')
datas = []
for g in glist:
    data = HeteroData()
    for ntype in "state county".split():
        data[ntype].x = g.nodes[ntype].data["feat"]
    for stype, etype, ttype in g.canonical_etypes:
        # src, dst = g.in_edges(g.nodes(ttype), etype=etype)
        # data[(stype,etype,ttype)].edge_index=torch.stack([src,dst]).long()
        data[(stype, etype, ttype)].edge_index = torch.stack(
            g.edges(etype=etype)
        ).long()
    datas.append(data)
torch.save(datas, f"covid.pt")

