import dgl
import torch
ei = torch.LongTensor([[0,1,3],[3,1,0]])
graph = dgl.graph((ei[0],ei[1])) #num_nodes=4
print(1)