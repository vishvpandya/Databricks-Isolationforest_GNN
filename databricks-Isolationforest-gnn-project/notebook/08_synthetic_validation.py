# Databricks notebook source
# MAGIC %pip install torch-geometric

# COMMAND ----------

# MAGIC %pip install torch

# COMMAND ----------

import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


# COMMAND ----------

class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(1, 16)
        self.conv2 = SAGEConv(16, 2)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        return x


# COMMAND ----------

model = GraphSAGE()
model.load_state_dict(
    torch.load("/Volumes/workspace/default/gnn_models/gnn_fraud_model.pt")
)
model.eval()

print("✅ Trained GNN model loaded")


# COMMAND ----------

NUM_NODES = 1000
FRAUD_RATIO = 0.3        # 30% fraud, 70% normal
NUM_EDGES = 3000

NUM_FRAUD = int(NUM_NODES * FRAUD_RATIO)
NUM_NORMAL = NUM_NODES - NUM_FRAUD

print("Fraud nodes:", NUM_FRAUD)
print("Normal nodes:", NUM_NORMAL)


# COMMAND ----------

fraud_nodes = list(range(NUM_FRAUD))
normal_nodes = list(range(NUM_FRAUD, NUM_NODES))

edges = []
labels = {}

# Assign labels
for n in fraud_nodes:
    labels[n] = 1
for n in normal_nodes:
    labels[n] = 0

# Fraud edges (dense, coordinated)
for _ in range(int(NUM_EDGES * 0.6)):
    src = random.choice(fraud_nodes)
    dst = random.choice(fraud_nodes)
    if src != dst:
        edges.append((src, dst))

# Normal edges (sparse, random)
for _ in range(int(NUM_EDGES * 0.4)):
    src = random.choice(normal_nodes)
    dst = random.choice(normal_nodes)
    if src != dst:
        edges.append((src, dst))


# COMMAND ----------

edge_index = torch.tensor(edges, dtype=torch.long).t()

# Degree-based node feature (same as training)
deg = Counter([e[0] for e in edges] + [e[1] for e in edges])

x = torch.tensor(
    [[deg.get(i, 0)] for i in range(NUM_NODES)],
    dtype=torch.float
)

y = torch.tensor(
    [labels[i] for i in range(NUM_NODES)],
    dtype=torch.long
)

data_syn = Data(x=x, edge_index=edge_index, y=y)

print("✅ Synthetic graph created")
print("Nodes:", data_syn.num_nodes)
print("Edges:", data_syn.num_edges)


# COMMAND ----------

# prob = F.softmax(out, dim=1)[:, 1]
# pred = (prob > 0.7).long()


# COMMAND ----------

with torch.no_grad():
    out = model(data_syn)
    # pred = out.argmax(dim=1)
    prob = F.softmax(out, dim=1)[:, 1]
    pred = (prob > 0.7).long()


# COMMAND ----------

y_true = y.numpy()
y_pred = pred.numpy()

cm = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print("📊 Synthetic Confusion Matrix:\n", cm)
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("Accuracy:", round(accuracy, 3))


# COMMAND ----------

