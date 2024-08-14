import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np

# Define node features for 10 person nodes with small random values
person_nodes = torch.randn((10, 1)) * 0.01  # Person nodes with small random values
age_nodes = torch.tensor([
    [0],  # Child
    [1],  # Adult
    [2]  # Elder
], dtype=torch.float)

sex_nodes = torch.tensor([
    [0],  # Male
    [1],  # Female
], dtype=torch.float)

ethnicity_nodes = torch.tensor([
    [0],  # White
    [1],  # Black
    [2]  # Asian
], dtype=torch.float)

# Combine all nodes
node_features = torch.cat([person_nodes, age_nodes, sex_nodes, ethnicity_nodes], dim=0)

# Observed counts from the cross table
observed_counts = {
    'Adult-Female-Black': 1,
    'Adult-Female-White': 2,
    'Adult-Male-Asian': 0,
    'Adult-Male-White': 0,
    'Child-Female-Black': 0,
    'Child-Female-White': 3,
    'Child-Male-Asian': 1,
    'Child-Male-White': 1,
    'Elder-Female-Black': 0,
    'Elder-Female-White': 1,
    'Elder-Male-Asian': 0,
    'Elder-Male-White': 1
}


# Calculate weights from observed data
def calculate_weights(observed_counts):
    total = sum(observed_counts.values())
    weights = {key: value / total for key, value in observed_counts.items()}
    return weights


weights = calculate_weights(observed_counts)


# Function to generate edge indexes randomly based on weights
def generate_edge_indexes(num_persons, weights):
    categories = ['Child', 'Adult', 'Elder']
    sexes = ['Male', 'Female']
    ethnicities = ['White', 'Black', 'Asian']

    age_weights = [weights[f'{age}-{sex}-{ethnicity}']
                   for age in categories for sex in sexes for ethnicity in ethnicities]
    sex_weights = [weights[f'{age}-{sex}-{ethnicity}']
                   for age in categories for sex in sexes for ethnicity in ethnicities]
    ethnicity_weights = [weights[f'{age}-{sex}-{ethnicity}']
                         for age in categories for sex in sexes for ethnicity in ethnicities]

    # Normalize weights
    age_weights = np.array(age_weights[:3])
    sex_weights = np.array(sex_weights[:2])
    ethnicity_weights = np.array(ethnicity_weights[:3])

    age_weights /= age_weights.sum()
    sex_weights /= sex_weights.sum()
    ethnicity_weights /= ethnicity_weights.sum()

    edge_index = [[], []]
    for person_id in range(num_persons):
        # Randomly select a characteristic ID based on the weights
        age_id = np.random.choice([10, 11, 12], p=age_weights)
        sex_id = np.random.choice([13, 14], p=sex_weights)
        ethnicity_id = np.random.choice([15, 16, 17], p=ethnicity_weights)

        edge_index[0].extend([person_id, person_id, person_id])
        edge_index[1].extend([age_id, sex_id, ethnicity_id])

    return torch.tensor(edge_index, dtype=torch.long)


# Generate edge indexes for 10 persons
edge_index = generate_edge_indexes(10, weights)
print(edge_index)

# Define the GraphSAGE model
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Ensure non-negative outputs from the first layer
        x = self.conv2(x, edge_index)
        x = torch.sigmoid(x)  # Ensure outputs are between 0 and 1
        return x


# Function to aggregate model predictions
def aggregate_predictions(predictions):
    categories = ['Child', 'Adult', 'Elder']
    sexes = ['Male', 'Female']
    ethnicities = ['White', 'Black', 'Asian']

    predicted_counts = {key: 0 for key in observed_counts.keys()}

    for i in range(predictions.size(0)):
        age_idx = torch.clamp(torch.round(predictions[i, 0] * 2), 0, 2).item()
        sex_idx = torch.clamp(torch.round(predictions[i, 1]), 0, 1).item()
        ethnicity_idx = torch.clamp(torch.round(predictions[i, 2] * 2), 0, 2).item()

        age = categories[int(age_idx)]
        sex = sexes[int(sex_idx)]
        ethnicity = ethnicities[int(ethnicity_idx)]

        key = f'{age}-{sex}-{ethnicity}'
        if key in predicted_counts:
            predicted_counts[key] += 1

    return predicted_counts


# Function to calculate custom loss based on observed and predicted counts
def calculate_custom_loss(predicted_counts, observed_counts):
    loss = 0.0
    for key in observed_counts:
        observed = observed_counts[key]
        predicted = predicted_counts.get(key, 0)
        loss += (observed - predicted) ** 2
    return torch.tensor(loss, dtype=torch.float, requires_grad=True)


# Hyperparameters
learning_rate = 0.01
hidden_channels = 16
num_epochs = 100

# Initialize model, optimizer, and loss function
model = GraphSAGEModel(in_channels=node_features.size(1), hidden_channels=hidden_channels, out_channels=3)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create data object
data = Data(x=node_features, edge_index=edge_index)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Generate new edge indexes based on weights
    data.edge_index = generate_edge_indexes(10, weights)

    out = model(data)[:10]  # Only take person nodes
    print(out)
    # Aggregate predictions
    predicted_counts = aggregate_predictions(out)

    # Calculate custom loss
    loss = calculate_custom_loss(predicted_counts, observed_counts)

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
