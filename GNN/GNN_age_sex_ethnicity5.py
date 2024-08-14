import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import CrossEntropyLoss
import pandas as pd

# Define node features for 10 person nodes with unique IDs as features
num_persons = 10
person_nodes = torch.arange(num_persons).view(num_persons, 1).float()  # Person nodes with unique IDs as features
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

# Create edges between person nodes and attribute nodes
edge_index = []
for i in range(num_persons):
    edge_index.append([i, 10 + i % 3])  # Connect person i to age node
    edge_index.append([i, 13 + i % 2])  # Connect person i to sex node
    edge_index.append([i, 15 + i % 3])  # Connect person i to ethnicity node

edge_index = torch.tensor(edge_index, dtype=torch.long).t()

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

# Create target tensors for age, sex, and ethnicity
age_map = {'Child': 0, 'Adult': 1, 'Elder': 2}
sex_map = {'Male': 0, 'Female': 1}
ethnicity_map = {'White': 0, 'Black': 1, 'Asian': 2}

# Initialize target tensors
y_age = torch.zeros(num_persons, dtype=torch.long)
y_sex = torch.zeros(num_persons, dtype=torch.long)
y_ethnicity = torch.zeros(num_persons, dtype=torch.long)

# Populate target tensors
person_idx = 0
for key, count in observed_counts.items():
    age, sex, ethnicity = key.split('-')
    for _ in range(count):
        y_age[person_idx] = age_map[age]
        y_sex[person_idx] = sex_map[sex]
        y_ethnicity[person_idx] = ethnicity_map[ethnicity]
        person_idx += 1

# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels_age, out_channels_sex, out_channels_ethnicity):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3_age = GCNConv(hidden_channels, out_channels_age)
        self.conv3_sex = GCNConv(hidden_channels, out_channels_sex)
        self.conv3_ethnicity = GCNConv(hidden_channels, out_channels_ethnicity)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        age_out = self.conv3_age(x, edge_index)
        sex_out = self.conv3_sex(x, edge_index)
        ethnicity_out = self.conv3_ethnicity(x, edge_index)
        return age_out, sex_out, ethnicity_out

# Initialize model, optimizer, and loss function
model = GNNModel(in_channels=node_features.size(1), hidden_channels=32, out_channels_age=3, out_channels_sex=2, out_channels_ethnicity=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn_age = CrossEntropyLoss()
loss_fn_sex = CrossEntropyLoss()
loss_fn_ethnicity = CrossEntropyLoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    age_out, sex_out, ethnicity_out = model(data)
    age_out = age_out[:num_persons]  # Only take person nodes' outputs
    sex_out = sex_out[:num_persons]
    ethnicity_out = ethnicity_out[:num_persons]

    loss_age = loss_fn_age(age_out, y_age)
    loss_sex = loss_fn_sex(sex_out, y_sex)
    loss_ethnicity = loss_fn_ethnicity(ethnicity_out, y_ethnicity)

    loss = loss_age + loss_sex + loss_ethnicity

    loss.backward()
    optimizer.step()

    # Calculate accuracy
    with torch.no_grad():
        age_pred = age_out.argmax(dim=1)
        sex_pred = sex_out.argmax(dim=1)
        ethnicity_pred = ethnicity_out.argmax(dim=1)

        age_accuracy = (age_pred == y_age).sum().item() / num_persons
        sex_accuracy = (sex_pred == y_sex).sum().item() / num_persons
        ethnicity_accuracy = (ethnicity_pred == y_ethnicity).sum().item() / num_persons

        # Calculate net accuracy
        net_accuracy = ((age_pred == y_age) & (sex_pred == y_sex) & (ethnicity_pred == y_ethnicity)).sum().item() / num_persons

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}, Age Accuracy: {age_accuracy:.4f}, Sex Accuracy: {sex_accuracy:.4f}, Ethnicity Accuracy: {ethnicity_accuracy:.4f}, Net Accuracy: {net_accuracy:.4f}')

# Get the final predictions after training
model.eval()
with torch.no_grad():
    age_out, sex_out, ethnicity_out = model(data)
    age_pred = age_out[:num_persons].argmax(dim=1)
    sex_pred = sex_out[:num_persons].argmax(dim=1)
    ethnicity_pred = ethnicity_out[:num_persons].argmax(dim=1)

# Convert final predictions to a dictionary
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

for i in range(num_persons):
    age = age_categories[age_pred[i]]
    sex = sex_categories[sex_pred[i]]
    ethnicity = ethnicity_categories[ethnicity_pred[i]]

    key = f"{age}-{sex}-{ethnicity}"
    if key in predicted_counts:
        predicted_counts[key] += 1

# Convert predicted counts to DataFrame
predicted_counts_df = pd.DataFrame(list(predicted_counts.items()), columns=['Category', 'Count'])

# Convert observed counts to DataFrame
observed_counts_df = pd.DataFrame(list(observed_counts.items()), columns=['Category', 'Count'])

# Print DataFrames
print("Predicted counts:")
print(predicted_counts_df)

print("Observed counts:")
print(observed_counts_df)
