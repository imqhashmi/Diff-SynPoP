import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import BCEWithLogitsLoss

# Define node features for 10 person nodes with unique IDs as features
person_nodes = torch.arange(10).view(10, 1).float()  # Person nodes with unique IDs as features
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

# Combine all nodes
node_features = torch.cat([person_nodes, age_nodes, sex_nodes, ethnicity_nodes], dim=0)

# Use the provided edge index
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Person IDs
    [11, 11, 11, 10, 10, 10, 10, 10, 12, 12, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 16, 15, 15, 15, 15, 15, 17, 15, 15, 15]  # Characteristic IDs
], dtype=torch.long)

# Create the data object with the given edge_index
data = Data(x=node_features, edge_index=edge_index)

# Observed data from the cross table
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

# Create binary target tensor (y) based on observed_counts
y = torch.zeros(10, 8)  # Now y has 8 columns
category_map = {
    'Child': 0,
    'Adult': 1,
    'Elder': 2,
    'Male': 0,
    'Female': 1,
    'White': 0,
    'Black': 1,
    'Asian': 2
}

# Updated target tensor creation based on observed_counts
person_idx = 0
for key, count in observed_counts.items():
    age, sex, ethnicity = key.split('-')
    age_idx = category_map[age]
    sex_idx = category_map[sex] + 3
    ethnicity_idx = category_map[ethnicity] + 5
    for _ in range(count):
        y[person_idx, age_idx] = 1
        y[person_idx, sex_idx] = 1
        y[person_idx, ethnicity_idx] = 1
        person_idx += 1

z = y

# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

# Initialize model, optimizer, and loss function
model = GNNModel(in_channels=node_features.size(1), hidden_channels=32, out_channels=8)  # Now output 8 features
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = BCEWithLogitsLoss()

# training = False
training = True
# Training loop
if training:
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data)[:10]  # Only take person nodes' outputs
        # out = model(data).squeeze()

        # Apply sigmoid to get edge probabilities
        edge_probs = torch.sigmoid(out)

        # Calculate binary cross-entropy loss
        loss = loss_fn(edge_probs, y)

        loss.backward()
        optimizer.step()

        # Calculate accuracy
        pred = edge_probs > 0.5
        correct = (pred == y.byte()).sum().item()
        accuracy = correct / (10 * 8)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')
