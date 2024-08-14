import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd

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

# Define static edges based on the cross table
# Person nodes: 0-9
# Age nodes: 10-12 (Child, Adult, Elder)
# Sex nodes: 13-14 (Male, Female)
# Ethnicity nodes: 15-17 (White, Black, Asian)
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Person IDs
    [11, 11, 11, 10, 10, 10, 10, 10, 12, 12, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 16, 15, 15, 15, 15, 15, 17, 15, 15, 15]  # Characteristic IDs
], dtype=torch.long)

# Create the data object
data = Data(x=node_features, edge_index=edge_index)

# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Ensure non-negative outputs from the first layer
        x = self.conv2(x, edge_index)
        return x

# Initialize model, optimizer, and loss function
model = GNNModel(in_channels=node_features.size(1), hidden_channels=16, out_channels=3)  # Output 3 features: age, sex, ethnicity
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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

# Differentiable custom aggregate function
def aggregate_predictions(predictions):
    categories = ['Child', 'Adult', 'Elder']
    sexes = ['Male', 'Female']
    ethnicities = ['White', 'Black', 'Asian']

    # Apply softmax to get probabilities for each category
    age_probs = F.softmax(predictions[:, 0].unsqueeze(1), dim=0)
    sex_probs = F.softmax(predictions[:, 1].unsqueeze(1), dim=0)
    ethnicity_probs = F.softmax(predictions[:, 2].unsqueeze(1), dim=0)

    # Calculate expected counts (differentiable)
    predicted_counts = {}
    for age_idx, age in enumerate(categories):
        for sex_idx, sex in enumerate(sexes):
            for eth_idx, ethnicity in enumerate(ethnicities):
                key = f"{age}-{sex}-{ethnicity}"
                predicted_counts[key] = torch.sum(
                    age_probs * sex_probs * ethnicity_probs
                )

    return predicted_counts

# Function to calculate the loss based on the observed and predicted counts
def calculate_loss(predicted_counts, observed_counts):
    loss = 0.0
    for key in observed_counts:
        observed = observed_counts[key]
        predicted = predicted_counts.get(key, torch.tensor(0.0))
        loss += (observed - predicted) ** 2
    return loss

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    out = model(data)[:10]  # Only take person nodes

    # Aggregate predictions
    predicted_counts = aggregate_predictions(out)

    # Calculate custom loss
    loss = calculate_loss(predicted_counts, observed_counts)

    loss.backward()

    # Print gradients for debugging
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Gradient for {name} at epoch {epoch}: {param.grad}")

    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
