import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import random

# Define node features for 10 person nodes with small random values
person_nodes = torch.randn((10, 1)) * 0.01  # Person nodes with small random values
age_nodes = torch.tensor([
    [0],  # Child
    [1],  # Adult
    [2]  # Elder
], dtype=torch.float)

sex_nodes = torch.tensor([
    [0],  # Male
    [1]  # Female
], dtype=torch.float)

ethnicity_nodes = torch.tensor([
    [0],  # White
    [1],  # Black
    [2]  # Asian
], dtype=torch.float)

# Combine all nodes
node_features = torch.cat([person_nodes, age_nodes, sex_nodes, ethnicity_nodes], dim=0)

# Define initial edges based on the cross table
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Person IDs
    [11, 11, 11, 10, 10, 10, 10, 10, 12, 12, 14, 14, 14, 14, 14, 14, 13, 13, 14, 14, 16, 15, 15, 15, 15, 15, 17, 15, 15,
     15]  # Characteristic IDs
], dtype=torch.long)


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
        x = torch.sigmoid(x)  # Ensure outputs are between 0 and 1
        return x


# Initialize model, optimizer, and loss function
model = GNNModel(in_channels=node_features.size(1), hidden_channels=16,
                 out_channels=3)  # Output 3 features: age, sex, ethnicity
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Create data object
data = Data(x=node_features, edge_index=edge_index)

# Hypothetical target population
target_population = torch.tensor([
    [1, 1, 1],  # Person 0: Adult, Female, Black
    [1, 1, 0],  # Person 1: Adult, Female, White
    [1, 1, 0],  # Person 2: Adult, Female, White
    [0, 1, 0],  # Person 3: Child, Female, White
    [0, 1, 0],  # Person 4: Child, Female, White
    [0, 1, 0],  # Person 5: Child, Female, White
    [0, 0, 2],  # Person 6: Child, Male, Asian
    [0, 0, 0],  # Person 7: Child, Male, White
    [2, 1, 0],  # Person 8: Elder, Female, White
    [2, 0, 0]  # Person 9: Elder, Male, White
], dtype=torch.float)

# Training loop
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()

    out = model(data)[:10]  # Only take person nodes

    # Print the model output for debugging
    # print(f'Output at epoch {epoch}: {out}')  # Debugging: Print the model output

    # Calculate MSE loss
    loss = loss_fn(out, target_population)

    loss.backward()

    # Print gradients for debugging
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Gradient for {name} at epoch {epoch}: {param.grad}")

    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Generate synthetic population
model.eval()
with torch.no_grad():
    synthetic_population = model(data)[:10]  # Only take person nodes
    rounded_population = torch.zeros_like(synthetic_population)
    rounded_population[:, 0] = torch.clamp(torch.round(synthetic_population[:, 0] * 2), 0, 2)  # Age: 0-2
    rounded_population[:, 1] = torch.clamp(torch.round(synthetic_population[:, 1]), 0, 1)  # Sex: 0-1
    rounded_population[:, 2] = torch.clamp(torch.round(synthetic_population[:, 2] * 2), 0, 2)  # Ethnicity: 0-2

    # Convert to DataFrame
    age_categories = ['Child', 'Adult', 'Elder']
    sex_categories = ['Male', 'Female']
    ethnicity_categories = ['White', 'Black', 'Asian']

    data = {
        'PersonID': list(range(10)),
        'Age': [age_categories[int(age)] for age in rounded_population[:, 0]],
        'Sex': [sex_categories[int(sex)] for sex in rounded_population[:, 1]],
        'Ethnicity': [ethnicity_categories[int(ethnicity)] for ethnicity in rounded_population[:, 2]]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    print(df)
