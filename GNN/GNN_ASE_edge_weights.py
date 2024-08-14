import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing

# Define the observed counts
observed_counts = {
    'Child-Male-White': 0,
    'Child-Male-Black': 0,
    'Child-Male-Asian': 0,
    'Child-Female-White': 2,
    'Child-Female-Black': 0,
    'Child-Female-Asian': 1,
    'Adult-Male-White': 1,
    'Adult-Male-Black': 2,
    'Adult-Male-Asian': 1,
    'Adult-Female-White': 0,
    'Adult-Female-Black': 0,
    'Adult-Female-Asian': 0,
    'Elder-Male-White': 0,
    'Elder-Male-Black': 0,
    'Elder-Male-Asian': 1,
    'Elder-Female-White': 1,
    'Elder-Female-Black': 1,
    'Elder-Female-Asian': 0
}

num_persons = 10

# Define the nodes for age, sex, and ethnicity categories
age_nodes = torch.tensor([
    [0],  # Child
    [1],  # Adult
    [2]   # Elder
], dtype=torch.float)

sex_nodes = torch.tensor([
    [0],  # Male
    [1]   # Female
], dtype=torch.float)

ethnicity_nodes = torch.tensor([
    [0],  # White
    [1],  # Black
    [2]   # Asian
], dtype=torch.float)

# Number of nodes for age, sex, and ethnicity
num_age_nodes = age_nodes.size(0)
num_sex_nodes = sex_nodes.size(0)
num_ethnicity_nodes = ethnicity_nodes.size(0)

# Define the edges based on the observed counts
edges = []
edge_weights = []

for i, (key, count) in enumerate(observed_counts.items()):
    age, sex, ethnicity = key.split('-')

    age_idx = ['Child', 'Adult', 'Elder'].index(age)
    sex_idx = ['Male', 'Female'].index(sex)
    ethnicity_idx = ['White', 'Black', 'Asian'].index(ethnicity)

    # Connect person node i to age, sex, and ethnicity nodes
    edges.append([i, num_persons + age_idx])
    edges.append([i, num_persons + num_age_nodes + sex_idx])
    edges.append([i, num_persons + num_age_nodes + num_sex_nodes + ethnicity_idx])

    # Weight the edges by the observed counts
    edge_weights.append(count)
    edge_weights.append(count)
    edge_weights.append(count)

# Convert edges and edge weights to tensors
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weights, dtype=torch.float)

# Create the data object for the GNN
x = torch.eye(num_persons + num_age_nodes + num_sex_nodes + num_ethnicity_nodes)
data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

print(edge_index)

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.fc(x)
        return x


# # Hyperparameters
# input_dim = num_persons + num_age_nodes + num_sex_nodes + num_ethnicity_nodes
# hidden_dim = 16
# output_dim = 8
# learning_rate = 0.01
# num_epochs = 200
#
# # Initialize the model, loss function, and optimizer
# model = GNNModel(input_dim, hidden_dim, output_dim)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # Training loop
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#
#     # Forward pass
#     out = model(data)
#
#     # Compute the loss
#     loss = criterion(out.squeeze(), edge_weight)
#
#     # Backward pass and optimization
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 100 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# model.eval()
# with torch.no_grad():
#     out = model(data)
#     print("Predicted Edge Weights:")
#     print(out.squeeze())
#     print("Actual Edge Weights:")
#     print(edge_weight)
