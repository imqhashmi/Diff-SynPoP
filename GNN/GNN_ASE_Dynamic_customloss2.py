import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
from torch.nn import CrossEntropyLoss
import random

# Define node features for 100 person nodes with unique IDs as features
num_persons = 5000
person_nodes = torch.arange(num_persons).view(num_persons, 1).float()

# Define nodes for age categories (Child, Adult, Elder)
age_nodes = torch.tensor([[0], [1], [2]], dtype=torch.float)

# Define nodes for sex categories (Male, Female)
sex_nodes = torch.tensor([[0], [1]], dtype=torch.float)

# Define nodes for ethnicity categories (White, Black, Asian)
ethnicity_nodes = torch.tensor([[0], [1], [2]], dtype=torch.float)

# Create mappings from categories to indices
age_map = {'Child': 0, 'Adult': 1, 'Elder': 2}
sex_map = {'Male': 0, 'Female': 1}
ethnicity_map = {'White': 0, 'Black': 1, 'Asian': 2}

def generate_observed_counts(num_persons):
    categories = [
        'Child-Male-White', 'Child-Male-Black', 'Child-Male-Asian',
        'Child-Female-White', 'Child-Female-Black', 'Child-Female-Asian',
        'Adult-Male-White', 'Adult-Male-Black', 'Adult-Male-Asian',
        'Adult-Female-White', 'Adult-Female-Black', 'Adult-Female-Asian',
        'Elder-Male-White', 'Elder-Male-Black', 'Elder-Male-Asian',
        'Elder-Female-White', 'Elder-Female-Black', 'Elder-Female-Asian'
    ]

    # Fictitious proportions based on UK census data
    proportions = [
        0.07, 0.01, 0.01,   # Child-Male-White, Child-Male-Black, Child-Male-Asian
        0.07, 0.01, 0.01,   # Child-Female-White, Child-Female-Black, Child-Female-Asian
        0.30, 0.02, 0.03,   # Adult-Male-White, Adult-Male-Black, Adult-Male-Asian
        0.31, 0.02, 0.03,   # Adult-Female-White, Adult-Female-Black, Adult-Female-Asian
        0.07, 0.01, 0.01,   # Elder-Male-White, Elder-Male-Black, Elder-Male-Asian
        0.07, 0.01, 0.01    # Elder-Female-White, Elder-Female-Black, Elder-Female-Asian
    ]

    # Ensure proportions sum to 1
    total = sum(proportions)
    proportions = [p / total for p in proportions]

    # Calculate counts based on proportions
    counts = [int(num_persons * p) for p in proportions]

    # Adjust to ensure total count matches num_persons
    diff = num_persons - sum(counts)
    counts[counts.index(max(counts))] += diff

    # Create the observed counts dictionary
    observed_counts = {category: count for category, count in zip(categories, counts)}
    return observed_counts


def generate_edge_index(num_persons):
    edge_index = []
    age_start_idx = num_persons
    sex_start_idx = age_start_idx + 3
    ethnicity_start_idx = sex_start_idx + 2

    for i in range(num_persons):
        age_category = random.choice([age_start_idx, age_start_idx + 1, age_start_idx + 2])
        sex_category = random.choice([sex_start_idx, sex_start_idx + 1])
        ethnicity_category = random.choice([ethnicity_start_idx, ethnicity_start_idx + 1, ethnicity_start_idx + 2])

        edge_index.append([i, age_category])
        edge_index.append([i, sex_category])
        edge_index.append([i, ethnicity_category])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

observed_counts = generate_observed_counts(num_persons)
node_features = torch.cat([person_nodes, age_nodes, sex_nodes, ethnicity_nodes], dim=0)
edge_index = generate_edge_index(num_persons)
data = Data(x=node_features, edge_index=edge_index)

y_age = torch.zeros(num_persons, dtype=torch.long)
y_sex = torch.zeros(num_persons, dtype=torch.long)
y_ethnicity = torch.zeros(num_persons, dtype=torch.long)

person_idx = 0
for key, count in observed_counts.items():
    age, sex, ethnicity = key.split('-')
    for _ in range(count):
        y_age[person_idx] = age_map[age]
        y_sex[person_idx] = sex_map[sex]
        y_ethnicity[person_idx] = ethnicity_map[ethnicity]
        person_idx += 1

# print("Target labels for age:", y_age)
# print("Target labels for sex:", y_sex)
# print("Target labels for ethnicity:", y_ethnicity)

# Define the enhanced GNN model with additional layers, batch norm, and dropout
class EnhancedGNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels_age, out_channels_sex, out_channels_ethnicity):
        super(EnhancedGNNModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.batch_norm2 = BatchNorm(hidden_channels)
        self.batch_norm3 = BatchNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5)
        self.conv4_age = SAGEConv(hidden_channels, out_channels_age)
        self.conv4_sex = SAGEConv(hidden_channels, out_channels_sex)
        self.conv4_ethnicity = SAGEConv(hidden_channels, out_channels_ethnicity)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.dropout(x)
        age_out = self.conv4_age(x, edge_index)
        sex_out = self.conv4_sex(x, edge_index)
        ethnicity_out = self.conv4_ethnicity(x, edge_index)
        return age_out, sex_out, ethnicity_out

# Custom loss function
def custom_loss_function(age_out, sex_out, ethnicity_out, y_age, y_sex, y_ethnicity):
    age_pred = age_out.argmax(dim=1)
    sex_pred = sex_out.argmax(dim=1)
    ethnicity_pred = ethnicity_out.argmax(dim=1)
    loss_age = F.cross_entropy(age_out, y_age)
    loss_sex = F.cross_entropy(sex_out, y_sex)
    loss_ethnicity = F.cross_entropy(ethnicity_out, y_ethnicity)
    total_loss = loss_age + loss_sex + loss_ethnicity
    return total_loss

# Initialize model, optimizer, and learning rate scheduler
model = EnhancedGNNModel(in_channels=node_features.size(1), hidden_channels=128, out_channels_age=3, out_channels_sex=2, out_channels_ethnicity=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

training = True
num_epochs = 400

if training:
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        age_out, sex_out, ethnicity_out = model(data)
        age_out = age_out[:num_persons]
        sex_out = sex_out[:num_persons]
        ethnicity_out = ethnicity_out[:num_persons]
        loss = custom_loss_function(age_out, sex_out, ethnicity_out, y_age, y_sex, y_ethnicity)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            age_pred = age_out.argmax(dim=1)
            sex_pred = sex_out.argmax(dim=1)
            ethnicity_pred = ethnicity_out.argmax(dim=1)
            age_accuracy = (age_pred == y_age).sum().item() / num_persons
            sex_accuracy = (sex_pred == y_sex).sum().item() / num_persons
            ethnicity_accuracy = (ethnicity_pred == y_ethnicity).sum().item() / num_persons
            net_accuracy = ((age_pred == y_age) & (sex_pred == y_sex) & (ethnicity_pred == y_ethnicity)).sum().item() / num_persons

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, Age Accuracy: {age_accuracy:.4f}, Sex Accuracy: {sex_accuracy:.4f}, Ethnicity Accuracy: {ethnicity_accuracy:.4f}, Net Accuracy: {net_accuracy:.4f}')

# Final evaluation after training
model.eval()
with torch.no_grad():
    age_out, sex_out, ethnicity_out = model(data)
    age_pred = age_out[:num_persons].argmax(dim=1)
    sex_pred = sex_out[:num_persons].argmax(dim=1)
    ethnicity_pred = ethnicity_out[:num_persons].argmax(dim=1)

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

print("Observed counts:")
print(observed_counts)
print("Predicted counts:")
print(predicted_counts)
