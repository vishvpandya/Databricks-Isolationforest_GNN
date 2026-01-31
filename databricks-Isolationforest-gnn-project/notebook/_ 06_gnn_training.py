# Databricks notebook source
# MAGIC %pip install --upgrade pip
# MAGIC # Recommended: Use Databricks Runtime ML, which comes with PyTorch pre-installed.
# MAGIC # If you must install manually, use supported PyPI versions:
# MAGIC %pip install torch torchvision
# MAGIC # torch-geometric and related packages are not available for aarch64 via pip, so they are commented out
# MAGIC # %pip install torch-geometric torch-scatter torch-sparse torch-cluster

# COMMAND ----------

# MAGIC %pip install torch 

# COMMAND ----------

# MAGIC %pip install torch-geometric

# COMMAND ----------

import pandas as pd
import torch

edges_df = spark.table("workspace.aml_fraud.gold_graph_edges") \
    .select("src", "dst", "label")

edges_pd = edges_df.toPandas()

edges_pd.head()


# COMMAND ----------

all_nodes = pd.unique(edges_pd[["src", "dst"]].values.ravel())

node_id_map = {node: idx for idx, node in enumerate(all_nodes)}

edges_pd["src_id"] = edges_pd["src"].map(node_id_map)
edges_pd["dst_id"] = edges_pd["dst"].map(node_id_map)

num_nodes = len(node_id_map)

print("Total nodes:", num_nodes)


# COMMAND ----------

edge_index = torch.tensor(
    [edges_pd["src_id"].values, edges_pd["dst_id"].values],
    dtype=torch.long
)


# COMMAND ----------

from collections import Counter

degree_count = Counter(edges_pd["src_id"]) + Counter(edges_pd["dst_id"])

x = torch.tensor(
    [[degree_count.get(i, 0)] for i in range(num_nodes)],
    dtype=torch.float
)


# COMMAND ----------

fraud_nodes = set(
    edges_pd.loc[edges_pd["label"] == 1, "src_id"]
).union(
    edges_pd.loc[edges_pd["label"] == 1, "dst_id"]
)

y = torch.tensor(
    [1 if i in fraud_nodes else 0 for i in range(num_nodes)],
    dtype=torch.long
)


# COMMAND ----------

from sklearn.model_selection import train_test_split

indices = list(range(num_nodes))
train_idx, test_idx = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=y
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
test_mask[test_idx] = True


# COMMAND ----------

from torch_geometric.data import Data

data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    test_mask=test_mask
)


# COMMAND ----------

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(1, 16)
        self.conv2 = SAGEConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# COMMAND ----------

model = GraphSAGE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


# COMMAND ----------

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

model.eval()
out = model(data)

pred = out.argmax(dim=1)

y_true = data.y[data.test_mask].cpu().numpy()
y_pred = pred[data.test_mask].cpu().numpy()

cm = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print("Confusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)


# COMMAND ----------

