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

# Define initial edges based on the cross table (used for parameterized weights)
edge_template = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Person IDs
    [11, 11, 11, 10, 10, 10, 10, 10, 12, 12, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 16, 15, 15, 15, 15, 15, 17, 15, 15, 15]  # Characteristic IDs
], dtype=torch.long)

# Create the data object with a placeholder edge_index
data = Data(x=node_features, edge_index=edge_template)

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
y = torch.zeros(10, 18)
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

for key, count in observed_counts.items():
    if count > 0:
        age, sex, ethnicity = key.split('-')
        age_idx = category_map[age]
        sex_idx = category_map[sex]
        ethnicity_idx = category_map[ethnicity]
        for i in range(count):
            y[i, age_idx] = 1
            y[i, sex_idx + 3] = 1
            y[i, ethnicity_idx + 5] = 1

print("Target tensor:", y)

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
model = GNNModel(in_channels=node_features.size(1), hidden_channels=32, out_channels=18)  # Increase hidden units to 32
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = BCEWithLogitsLoss()


# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    out = model(data)[:10]  # Only take person nodes' outputs

    # Apply sigmoid to get edge probabilities
    edge_probs = torch.sigmoid(out)

    # Calculate binary cross-entropy loss
    loss = loss_fn(edge_probs, y)

    loss.backward()
    optimizer.step()

    # Calculate accuracy
    pred = edge_probs > 0.5
    correct = (pred == y.byte()).sum().item()
    accuracy = correct / (10 * 18)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')

# Print final edge probabilities
print("Final edge probabilities:")
final_edge_probs = torch.sigmoid(model(data)[:10]).detach().numpy()

# Convert final edge probabilities to a dictionary
predicted_counts = {
    'Child-Male-White': 0,
    'Child-Male-Black': 0,
    'Child-Male-Asian': 0,
    'Child-Female-White': 0,
    'Child-Female-Black': 0,
    'Child-Female-Asian': 0,
    'Adult-Male-White': 0,
    'Adult-Male-Black': 0,
    'Adult-Male-Asian': 0,
    'Adult-Female-White': 0,
    'Adult-Female-Black': 0,
    'Adult-Female-Asian': 0,
    'Elder-Male-White': 0,
    'Elder-Male-Black': 0,
    'Elder-Male-Asian': 0,
    'Elder-Female-White': 0,
    'Elder-Female-Black': 0,
    'Elder-Female-Asian': 0
}

age_categories = ['Child', 'Adult', 'Elder']
sex_categories = ['Male', 'Female']
ethnicity_categories = ['White', 'Black', 'Asian']

for i in range(10):
    age_prob = final_edge_probs[i, 0:3]
    sex_prob = final_edge_probs[i, 3:5]
    ethnicity_prob = final_edge_probs[i, 5:8]

    age = age_categories[age_prob.argmax()]
    sex = sex_categories[sex_prob.argmax()]
    ethnicity = ethnicity_categories[ethnicity_prob.argmax()]

    key = f"{age}-{sex}-{ethnicity}"
    if key in predicted_counts:
        predicted_counts[key] += 1

# Compare predicted counts with observed counts
print("Predicted counts:")
print(predicted_counts)

print("Observed counts:")
print(observed_counts)
