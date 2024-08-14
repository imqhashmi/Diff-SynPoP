import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Simplified example node features for persons and households
person_features = torch.tensor([
    [25, 0, 1],  # Person 1: 25 years old, male, ethnicity 1
    [30, 1, 2],  # Person 2: 30 years old, female, ethnicity 2
    [22, 0, 3]   # Person 3: 22 years old, male, ethnicity 3
], dtype=torch.float)

household_features = torch.tensor([
    [2, 1, 3],  # Household 1: 2 members, ethnicity 1, composition 3
    [3, 2, 4]   # Household 2: 3 members, ethnicity 2, composition 4
], dtype=torch.float)

# Combine person and household features
node_features = torch.cat([person_features, household_features], dim=0)

# Example edge index
# Edges between persons (nodes 0, 1, 2) and households (nodes 3, 4)
edge_index = torch.tensor([
    [0, 1, 2, 3, 4],
    [3, 3, 4, 0, 1]
], dtype=torch.long)

# Define target attributes (same structure as node features, typically from observed data)
target_person_features = torch.tensor([
    [26, 0, 1],  # Adjusted attributes for person 1
    [31, 1, 2],  # Adjusted attributes for person 2
    [23, 0, 3]   # Adjusted attributes for person 3
], dtype=torch.float)

target_household_features = torch.tensor([
    [2, 1, 3],  # Adjusted attributes for household 1
    [3, 2, 4]   # Adjusted attributes for household 2
], dtype=torch.float)

# Combine target person and household features
target_features = torch.cat([target_person_features, target_household_features], dim=0)


# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize model, optimizer, and loss function
model = GNNModel(in_channels=node_features.size(1), hidden_channels=16, out_channels=node_features.size(1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Create data object
data = Data(x=node_features, edge_index=edge_index)

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, target_features)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Generate synthetic population
model.eval()
with torch.no_grad():
    synthetic_population = model(data)
    print("Synthetic Population Features:\n", synthetic_population)
