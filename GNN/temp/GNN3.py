import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Distributions
age_dist = {'Child': 0.25, 'Adult': 0.6, 'Elder': 0.15}
sex_dist = {'Male': 0.5, 'Female': 0.5}
ethnicity_dist = {'White': 0.6, 'Black': 0.2, 'Asian': 0.2}
religion_dist = {'Christian': 0.5, 'Jew': 0.1, 'Muslim': 0.4}

n = 100

# Create a directed graph
G = nx.DiGraph()


# Helper function to sample based on distribution
def sample_distribution(distribution, size):
    return np.random.choice(list(distribution.keys()), size=size, p=list(distribution.values()))


# Generate nodes for each attribute
ages = {f'A{i + 1}': age for i, age in enumerate(sample_distribution(age_dist, n))}
sexs = {f'S{i + 1}': sex for i, sex in enumerate(sample_distribution(sex_dist, n))}
ethnicities = {f'E{i + 1}': eth for i, eth in enumerate(sample_distribution(ethnicity_dist, n))}
religions = {f'R{i + 1}': rel for i, rel in enumerate(sample_distribution(religion_dist, n))}

# Add attribute nodes to the graph
for key, value in ages.items():
    G.add_node(key, type='age', label=value)

for key, value in sexs.items():
    G.add_node(key, type='sex', label=value)

for key, value in ethnicities.items():
    G.add_node(key, type='ethnicity', label=value)

for key, value in religions.items():
    G.add_node(key, type='religion', label=value)

for i in range(n):
    G.add_node(f'P{i}', type='person', label=f'P{i}')

# Add edges between person and attribute nodes
for i in range(n):
    G.add_edge(f'P{i}', f'A{i + 1}')
    G.add_edge(f'P{i}', f'S{i + 1}')
    G.add_edge(f'P{i}', f'E{i + 1}')
    G.add_edge(f'P{i}', f'R{i + 1}')

import pyvis
from pyvis.network import Network
nt = Network(height='750px', width='100%')
nt.from_nx(G)
nt.show('nx.html', notebook=False)
# # Encode labels as integers
# label_encoder = LabelEncoder()
#
# # Encode age, sex, ethnicity, and religion labels
# age_labels = label_encoder.fit_transform(list(ages.values()))
# sex_labels = label_encoder.fit_transform(list(sexs.values()))
# ethnicity_labels = label_encoder.fit_transform(list(ethnicities.values()))
# religion_labels = label_encoder.fit_transform(list(religions.values()))
#
# # Combine labels into node features
# node_features = np.vstack([age_labels, sex_labels, ethnicity_labels, religion_labels]).T
#
# # Create PyTorch Geometric data object
# edge_index = torch.tensor(list(G.edges)).t().contiguous()
# x = torch.tensor(node_features, dtype=torch.float)
# data = Data(x=x, edge_index=edge_index)
#
#
# # Define a simple MLP model using PyTorch Geometric
# class MLP(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MLP, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.lin = Linear(hidden_dim, output_dim)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = global_mean_pool(x, torch.arange(data.num_nodes))
#         x = self.lin(x)
#         return F.log_softmax(x, dim=1)
#
#
# # Define model, loss function and optimizer
# model = MLP(input_dim=4, hidden_dim=16, output_dim=len(label_encoder.classes_))
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss_fn = torch.nn.CrossEntropyLoss()
#
#
# # Training the model
# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data)
#     loss = loss_fn(out, torch.tensor(age_labels, dtype=torch.long))
#     loss.backward()
#     optimizer.step()
#     return loss.item()
#
#
# # Train the model for 200 epochs
# for epoch in range(200):
#     loss = train()
#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Loss: {loss}')
#
# # Make predictions
# model.eval()
# with torch.no_grad():
#     pred = model(data).argmax(dim=1)
#
#
# # Evaluate how well the predicted attributes fit the cross tables
# def evaluate_predictions(predictions, cross_table):
#     # Convert predictions to categorical labels
#     predicted_labels = label_encoder.inverse_transform(predictions)
#     predicted_counts = dict(zip(*np.unique(predicted_labels, return_counts=True)))
#
#     # Compare predicted counts with cross table counts
#     total_diff = 0
#     for key, value in cross_table.items():
#         predicted_value = predicted_counts.get(key, 0)
#         total_diff += abs(predicted_value - value)
#
#     return total_diff
#
#
# # Evaluate the model
# cross_table1_diff = evaluate_predictions(pred, crosstable1)
# cross_table2_diff = evaluate_predictions(pred, crosstable2)
#
# print(f'Cross Table 1 Difference: {cross_table1_diff}')
# print(f'Cross Table 2 Difference: {cross_table2_diff}')
