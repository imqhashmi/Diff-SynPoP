import numpy as np
import networkx as nx
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Distributions
age_dist = {'Child': 0.25, 'Adult': 0.6, 'Elder': 0.15}
sex_dist = {'Male': 0.5, 'Female': 0.5}
ethnicity_dist = {'White': 0.6, 'Black': 0.2, 'Asian': 0.2}
religion_dist = {'C': 0.5, 'J': 0.1, 'M': 0.4}

# Cross tables
crosstable1 = {'Adult-Female-Asian': 4, 'Adult-Female-Black': 3, 'Adult-Female-White': 21, 'Adult-Male-Asian': 8,
               'Adult-Male-Black': 5, 'Adult-Male-White': 22, 'Child-Female-Asian': 2, 'Child-Female-Black': 2,
               'Child-Female-White': 3, 'Child-Male-Asian': 3, 'Child-Male-Black': 2, 'Child-Male-White': 5,
               'Elder-Female-Asian': 1, 'Elder-Female-Black': 2, 'Elder-Female-White': 6, 'Elder-Male-Asian': 2,
               'Elder-Male-Black': 3, 'Elder-Male-White': 6}
crosstable2 = {'Adult-Female-C': 10, 'Adult-Female-J': 6, 'Adult-Female-M': 5, 'Adult-Male-C': 19, 'Adult-Male-J': 4,
               'Adult-Male-M': 13, 'Child-Female-C': 6, 'Child-Female-J': 2, 'Child-Female-M': 5, 'Child-Male-C': 6,
               'Child-Male-J': 1, 'Child-Male-M': 6, 'Elder-Female-C': 5, 'Elder-Female-J': 1, 'Elder-Female-M': 5,
               'Elder-Male-C': 3, 'Elder-Male-J': 0, 'Elder-Male-M': 3}


# Function to generate sample based on distribution
def weighted_sample(distribution):
    return np.random.choice(list(distribution.keys()), p=list(distribution.values()))


# Function to create a person with attributes
def create_person():
    age = weighted_sample(age_dist)
    sex = weighted_sample(sex_dist)
    ethnicity = weighted_sample(ethnicity_dist)
    religion = weighted_sample(religion_dist)
    return age, sex, ethnicity, religion


# Generate 100 persons
persons = [create_person() for _ in range(100)]


# Create a GNN
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(4, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc = nn.Linear(32, 4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x


# Function to create graph data for the GNN
def create_graph_data(persons):
    edge_index = []
    x = []
    person_dict = {'Child': 0, 'Adult': 1, 'Elder': 2, 'Male': 3, 'Female': 4, 'White': 5, 'Black': 6, 'Asian': 7,
                   'C': 8, 'J': 9, 'M': 10}

    for i, person in enumerate(persons):
        age, sex, ethnicity, religion = person
        person_node = [0] * 11
        person_node[person_dict[age]] = 1
        person_node[person_dict[sex]] = 1
        person_node[person_dict[ethnicity]] = 1
        person_node[person_dict[religion]] = 1
        x.append(person_node)
        edge_index.append([i, person_dict[age]])
        edge_index.append([i, person_dict[sex]])
        edge_index.append([i, person_dict[ethnicity]])
        edge_index.append([i, person_dict[religion]])

    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data


# Function to calculate loss based on cross tables
def calculate_loss(output, persons):
    loss = 0
    for i, person in enumerate(persons):
        age, sex, ethnicity, religion = person
        key1 = f"{age}-{sex}-{ethnicity}"
        key2 = f"{age}-{sex}-{religion}"
        if key1 in crosstable1:
            loss += (output[i][0] - crosstable1[key1]) ** 2
        if key2 in crosstable2:
            loss += (output[i][1] - crosstable2[key2]) ** 2
    return loss


# Initialize the GNN
gnn = GNN()
optimizer = optim.Adam(gnn.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Create graph data
data = create_graph_data(persons)

# Train the GNN
for epoch in range(200):
    optimizer.zero_grad()
    output = gnn(data)
    loss = calculate_loss(output, persons)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Final output
final_output = gnn(data)
print(final_output)
