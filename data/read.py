import numpy as np
data = np.load('fin/dgraphfin.npz')
print(data.files)
# print(1)
# unique_elements, counts = np.unique( data['edge_timestamp'], return_counts=True)
# for element, count in zip(unique_elements, counts):
#     if count > 1:
#         print(f"元素 {element} 重复了 {count} 次。")
x = data['x']  # 17维节点特征
y = data['y']  # 节点标签
edge_index = data['edge_index']  # 边的索引
edge_type = data['edge_type']  # 边的类型
edge_timestamp = data['edge_timestamp']  # 边的时间戳
train_mask = data['train_mask']  # 训练掩码
valid_mask = data['valid_mask']  # 验证掩码
test_mask = data['test_mask']  # 测试掩码

max_timestamp = max(edge_timestamp)
time_window = 100

# 筛选出前100天的边
valid_edges_mask = edge_timestamp >= (max_timestamp - time_window)
edge_index_filtered = edge_index[valid_edges_mask]
edge_type_filtered = edge_type[valid_edges_mask]
edge_timestamp_filtered = edge_timestamp[valid_edges_mask]

# 2. 构建异质图
# 获取唯一的节点ID
unique_node_ids = np.unique(edge_index_filtered)

graphs = []
for day in range(time_window):
    day_edges_mask = edge_timestamp_filtered == (max_timestamp - day)
    day_edges = edge_index_filtered[day_edges_mask]
    day_edge_types = edge_type_filtered[day_edges_mask]
