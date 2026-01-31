# Databricks notebook source
# Install the torch package, required for the notebook to run without ModuleNotFoundError
%pip install torch

# COMMAND ----------

import pandas as pd
import torch
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


# COMMAND ----------

edges_df = spark.table("workspace.aml_fraud.gold_graph_edges_full") \
    .select("src", "dst", "label")

print("Total edges:", edges_df.count())
display(edges_df.limit(5))


# COMMAND ----------

# Fraud nodes (from labels)
fraud_nodes = edges_df.filter(col("label") == 1) \
    .select(col("src").alias("node_id")) \
    .union(edges_df.filter(col("label") == 1).select(col("dst").alias("node_id"))) \
    .distinct()

# Normal nodes (CONTROL GROUP)
normal_nodes = edges_df.filter(col("label") == 0) \
    .select(col("src").alias("node_id")) \
    .union(edges_df.filter(col("label") == 0).select(col("dst").alias("node_id"))) \
    .distinct() \
    .sample(fraction=0.3, seed=42)   # 👈 control size (adjustable)

eval_nodes = fraud_nodes.union(normal_nodes).distinct()

print("Fraud nodes:", fraud_nodes.count())
print("Control normal nodes:", normal_nodes.count())
print("Total eval nodes:", eval_nodes.count())


# COMMAND ----------

eval_edges = edges_df.join(
    eval_nodes.withColumnRenamed("node_id", "src_id"),
    edges_df.src == col("src_id"),
    "inner"
)

print("Evaluation edges:", eval_edges.count())


# COMMAND ----------

# LIMIT the size BEFORE bringing to driver
edges_sampled = eval_edges.sample(fraction=0.1, seed=42)  # start with 10%

print("Sampled edges:", edges_sampled.count())

edges_pd = edges_sampled.toPandas()
edges_pd.head()



# COMMAND ----------

all_nodes = pd.unique(edges_pd[["src", "dst"]].values.ravel())
node_map = {node: i for i, node in enumerate(all_nodes)}

edges_pd["src_id"] = edges_pd["src"].map(node_map)
edges_pd["dst_id"] = edges_pd["dst"].map(node_map)

num_nodes = len(node_map)
print("Total nodes for evaluation:", num_nodes)


# COMMAND ----------

edge_index = torch.tensor(
    [edges_pd["src_id"].values, edges_pd["dst_id"].values],
    dtype=torch.long
)


# COMMAND ----------

from collections import Counter

deg = Counter(edges_pd["src_id"]) + Counter(edges_pd["dst_id"])

x = torch.tensor(
    [[deg.get(i, 0)] for i in range(num_nodes)],
    dtype=torch.float
)


# COMMAND ----------

fraud_node_ids = set(
    edges_pd.loc[edges_pd["label"] == 1, "src_id"]
).union(
    edges_pd.loc[edges_pd["label"] == 1, "dst_id"]
)

y = torch.tensor(
    [1 if i in fraud_node_ids else 0 for i in range(num_nodes)],
    dtype=torch.long
)

print("Fraud nodes:", int(y.sum().item()))
print("Normal nodes:", num_nodes - int(y.sum().item()))


# COMMAND ----------

indices = list(range(num_nodes))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.3,
    random_state=42,
    stratify=y
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
test_mask[test_idx] = True


# COMMAND ----------

# MAGIC %pip install torch-geometric

# COMMAND ----------

from torch_geometric.data import Data

data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    test_mask=test_mask
)

# from collections import Counter
# import torch

# # degree-based node feature
# deg = Counter(edges_pd["src_id"]) + Counter(edges_pd["dst_id"])

# x = torch.tensor(
#     [[deg.get(i, 0)] for i in range(num_nodes)],
#     dtype=torch.float
# )

# print("Node feature tensor shape:", x.shape)



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

import os

# Replaced DBFS directory creation with Unity Catalog volume creation using Spark SQL
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.gnn_models")
print("✅ Unity Catalog volume 'gnn_models' ready")

# COMMAND ----------

import torch
# Save model to Unity Catalog volume (recommended for shared clusters)
torch.save(model.state_dict(), "/Volumes/workspace/default/gnn_models/gnn_fraud_model.pt")
print("✅ Model saved to Unity Catalog volume")

# COMMAND ----------

