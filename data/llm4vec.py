from tqdm import tqdm
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model_name = "../../../LLM_models/llama-2-7b-hf"  # 模型路径或名称

def llm4vec(node_descriptions):
    # model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda:0')
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    # model = torch.nn.DataParallel(model)
    model = model.to('cuda:0')
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = 8
    all_node_feature_vectors = []
    for i in tqdm(range(0, len(node_descriptions), batch_size), desc="Processing Nodes"):
        batch_descriptions = node_descriptions[i:i + batch_size]
        inputs = tokenizer(batch_descriptions, padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda:0')
        with torch.no_grad():  # 在推理时禁用梯度计算
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        all_node_feature_vectors.append(last_hidden_states.mean(dim=1))

    return all_node_feature_vectors


import os
import os.path as osp
import torch
import torch as th
import numpy as np
from collections import Counter
from torch_geometric.transforms import ToUndirected
import gensim
from gensim.models import Word2Vec
from torch_geometric.data import HeteroData

fnames = ["Database", "Data Mining", "Medical Informatics", "Theory", "Visualization"]

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# dataroot = os.path.join(CUR_DIR, "../data/")
dataroot = osp.join(CUR_DIR, "Aminer")
datafiles = [f"{dataroot}/{name}.txt" for name in fnames]
word2vec_file = f"{dataroot}/abs2vec"
processed_datafile = f"{dataroot}/processed"

def parse(datafile):
    field = os.path.split(datafile)[-1]
    field = os.path.splitext(field)[0]  # 去掉后缀名.txt
    with open(datafile, "r") as file:
        lines = file.readlines()
    lines[0].split("\t")
    papers = []
    for line in lines:
        (venue, title, authors, year, abstract) = line.split("\t")
        try:
            year = int(year)
            paper = (venue, title, authors, year, abstract, field)
            papers.append(paper)
        except Exception as e:
            print(e)
        # papers.append(paper)
    papers = np.array(papers)
    return papers

def sen2vec(sentences, vector_size=32):
    """use gensim.word2vec to generate wordvecs, and average as sentence vectors.
    if exception happens use zero embedding.
    @ params sentences : list of sentence
    @ params vector_size
    @ return : sentence embedding
    """
    sentences = [list(gensim.utils.tokenize(a, lower=True)) for a in sentences]
    sentences
    # vector_size = 32
    model = Word2Vec(sentences, vector_size=vector_size, min_count=1)
    print("word2vec done")
    embs = []
    for s in sentences:
        try:
            emb = model.wv[s]
            emb = np.mean(emb, axis=0)
        except Exception as e:
            print(e)
            emb = np.zeros(vector_size)
        embs.append(emb)
    embs = np.stack(embs)
    print(f"emb shape : {embs.shape}")
    return embs

class AminerDataset:
    """Aminer CrossDomain Dataset
    we use gensim.word2vec
    """

    def __init__(
        self, undirected=True,  word2vec_size=32, device="auto"):

        self.device = device
        processed = f"{processed_datafile}-{word2vec_size}.pt"
        if osp.exists(processed):
            # print(f'loading {processed}')
            # dataset = torch.load(processed)
            dataset = self.preprocess(word2vec_size)
            torch.save(dataset, processed)

        else:
            dataset = self.preprocess(word2vec_size)
            torch.save(dataset, processed)

        if undirected:
            dataset = ToUndirected()(dataset)

        self.dataset = dataset

    def times(self):
        return sorted(
            list(Counter(self.dataset.time_dict["paper"].squeeze().numpy()).keys())
        )


    def preprocess(self, word2vec_size=32):
        papers = []
        for file in datafiles:
            paper = parse(file)
            papers.append(paper)

        # 统计每个年份有多少paper
        for i, paper in enumerate(papers):
            print(fnames[i])
            print(Counter(paper[:, 3]))
        papers = np.concatenate(papers)

        #filter
        papers = papers[np.array([row[2] not in ("",None) for row in papers])]

        Counter(papers[:, 3])  # year
        Counter(papers[:, 0])  # venue

        authors = []
        for paper in papers:
            authors.extend(paper[2].split(","))
        len(authors)  # authors

        # do mapping
        def map2id(l):
            return dict(zip(l, range(len(l))))

        def sorteddict(x, min=True):
            """return dict sorted by values
            @params x: a dict
            @params min : whether from small to large.
            """
            if min:
                return dict(sorted(x.items(), key=lambda item: item[1]))
            else:
                return dict(sorted(x.items(), key=lambda item: item[1])[::-1])

        vid2vname = list(Counter(papers[:, 0]).keys())
        vname2vid = map2id(vid2vname)
        vname2fname = dict(zip(papers[:,0],papers[:,-1]))

        authors = []
        for paper in papers:
            authors.extend(paper[2].split(","))
        aid2aname = list(sorteddict(Counter(authors), min=False).keys())
        aname2aid = map2id(aid2aname)


        yid2yname = sorted(list(map(int, Counter(papers[:, 3]).keys())))
        yname2yid = map2id(yid2yname)
        print("tid2tname:", yname2yid)

        fid2fname = sorted(list(Counter(papers[:, 5]).keys()))
        fname2fid = map2id(fid2fname)

        # venue link
        e_pv = []
        for i, vname in enumerate(papers[:, 0]):
            e_pv.append([i, vname2vid[vname]])
        e_pv = th.LongTensor(np.array(e_pv)).T

        # author link
        e_pa = []
        for i, anames in enumerate(papers[:, 2]):
            for aname in anames.split(","):
                e_pa.append([i, aname2aid[aname]])
        e_pa = th.LongTensor(np.array(e_pa)).T

        # title; we do not use
        x_title = papers[:, 1]

        # years
        x_year = th.LongTensor(list(map(lambda x: yname2yid[int(x)], papers[:, 3])))


        # field
        x_field = th.LongTensor(list(map(lambda x: fname2fid[x], papers[:, 5])))


        # abstract

        # x_abstract = papers[:, 4]
        node_descriptions = []
        for row in papers:
            conference = row[0]
            title = row[1]
            authors = row[2]
            year = row[3]
            abstract = row[4].replace(".\n","")
            field = row[5]

            # 创建符合格式的字符串
            description = (f"Feature node. Conference: {conference}; "
                           f"Title: {title}; "
                           f"Authors: {authors}; "
                           f"Year: {year}; "
                           f"Abstract: {abstract};"
                           f"Field: {field};")

            # 将每个节点描述添加到列表中
            node_descriptions.append(description)

        # emb_file = f"{word2vec_file}-{word2vec_size}.npy"
        # if False: #os.path.exists(emb_file)
        #     print(f"loading {emb_file}")
        #     emb_abs = np.load(emb_file)
        # else:
        #     print(f"generating {emb_file}")
        #     emb_abs = llm4vec(node_descriptions)
        #     torch(emb_abs,emb_file)

        emb_abs = torch.load('/media/yjz/wyl/ijcai2025/data/Aminer/abs2vec-4096.npy')
        emb_abs = torch.cat(emb_abs, dim=0)
        # emb_field
        # x_field = papers[:, 4]
        emb_field = dict(zip(fid2fname,sen2vec(fid2fname, vector_size=word2vec_size)))
        emb_field = np.array([emb_field[vname2fname[i]] for i in vid2vname])

        # emb_author
        emb_author = np.array(sen2vec(aid2aname, vector_size=word2vec_size))

        # author
        num_author = len(set(e_pa[1, :].numpy()))
        x_author = torch.arange(num_author)


        # venue
        num_venue = len(set(e_pv[1, :].numpy()))
        x_venue = torch.arange(num_venue)


        data = HeteroData()
        data["paper", "published", "venue"].edge_index = e_pv
        data["paper", "written", "author"].edge_index = e_pa
        data["paper"].x = emb_abs.to(torch.float32).cpu()
        data["paper"].y = x_field
        data["paper"].time = x_year.unsqueeze(-1)
        # data["author"].x = torch.FloatTensor(emb_author)
        data["author"].x = x_author.unsqueeze(-1)
        data["venue"].x = x_venue.unsqueeze(-1)
        # data["venue"].x = torch.FloatTensor(emb_field)
        data["paper"].num_nodes = data["published"].edge_index.shape[1]

        data["published"].edge_time = data["paper"].time.index_select(0, e_pv[0, :])
        data["written"].edge_time = data["paper"].time.index_select(0, e_pa[0, :])

        info_dict = {
            "vid2vname": vid2vname,
            "vname2vid": vname2vid,
            "aid2aname": aid2aname,
            "aname2aid": aname2aid,
            "yid2yname": yid2yname,
            "yname2yid": yname2yid,
        }
        return data


if __name__ == "__main__":
    time_window = 5
    shuffle = True
    test_full = False
    is_dynamic = False
    dataset = AminerDataset(
        undirected=True,
        word2vec_size = 4096)


