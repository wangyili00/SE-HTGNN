import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "Meta-Llama-3-8B-Instruct"  # 模型路径或名称
device = torch.device('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to(device)
dataset = "Covid19"
# 输入文本
if dataset == "Aminer":
    input_text = ["Academic paper is a type of node in the academic dynamic graph. This type of nodes are connected to Author-type nodes and Venues-type nodes.",
                  "Author is a type of node in the academic dynamic graph. This type of nodes are connected to Paper-type nodes.",
                  "Venue is a type of node in the academic dynamic graph. This type of nodes are connected to Paper-type nodes."]
elif dataset == "Covid19":
    input_text = [
        "State is a type of node in the epidemic dynamic graph. This type of nodes are typically connected to state-type nodes and county-type nodes.",
        "County is a type of node in the epidemic dynamic graph. This type of nodes is typically connected to county-type nodes and state-type nodes."
    ]

# elif dataset == "ogbn":
#     input_text = [
#                   "Paper: An academic work written by authors, which may cite other papers and belong to a specific field of study.",
#                   "Author: A researcher who writes papers and is affiliated with an institution.",
#                   "Institution: An organization that supports and houses authors.",
#                   "Field_of_Study: An academic area or discipline that includes various related papers."
#     ]
# elif dataset == "yelp":
#     input_text = ["Users of the business review net Yelp. They may review or tip businesses.",
#                   "Businesses of the business review net Yelp. They may be reviewed or tipped by users."]

request = " Please output a summary of the information about this node type in the following format: {Introduction:,Relevant relations:}."

all_node_feature_vectors = []
for text in input_text:
    inputs = text + request
    inputs = tokenizer(inputs, return_tensors="pt").to(device)
    with torch.no_grad():  # 在推理时禁用梯度计算
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden_states = outputs.hidden_states[-1]
    all_node_feature_vectors.append(last_hidden_states.mean(dim=1))
# LLM_feature = {"paper": all_node_feature_vectors[0].to(torch.float32).cpu(),
#                "author":all_node_feature_vectors[1].to(torch.float32).cpu(),
#                "venue": all_node_feature_vectors[2].to(torch.float32).cpu()}

# LLM_feature = {"user": all_node_feature_vectors[0].to(torch.float32).cpu(),
#                "item":all_node_feature_vectors[1].to(torch.float32).cpu(),} YELP\

# LLM_feature = {"paper": all_node_feature_vectors[0].to(torch.float32).cpu(),
#                "author":all_node_feature_vectors[1].to(torch.float32).cpu(),
#                "institution": all_node_feature_vectors[2].to(torch.float32).cpu(),
#                 "field_of_study":all_node_feature_vectors[3].to(torch.float32).cpu()
#                }
LLM_feature = {"state": all_node_feature_vectors[0].to(torch.float32).cpu(),
               "county":all_node_feature_vectors[1].to(torch.float32).cpu(),}

torch.save(LLM_feature, f'{dataset}/LLM_feature_Llama-3-new.pt')
print("over")



